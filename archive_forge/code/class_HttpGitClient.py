from contextlib import closing
from io import BytesIO, BufferedReader
import logging
import os
import select
import socket
import subprocess
import sys
from typing import Optional, Dict, Callable, Set
from urllib.parse import (
import dulwich
from dulwich.config import get_xdg_config_home_path
from dulwich.errors import (
from dulwich.protocol import (
from dulwich.pack import (
from dulwich.refs import (
class HttpGitClient(GitClient):

    def __init__(self, base_url, dumb=None, pool_manager=None, config=None, username=None, password=None, **kwargs):
        self._base_url = base_url.rstrip('/') + '/'
        self._username = username
        self._password = password
        self.dumb = dumb
        if pool_manager is None:
            self.pool_manager = default_urllib3_manager(config)
        else:
            self.pool_manager = pool_manager
        if username is not None:
            credentials = '%s:%s' % (username, password)
            import urllib3.util
            basic_auth = urllib3.util.make_headers(basic_auth=credentials)
            self.pool_manager.headers.update(basic_auth)
        GitClient.__init__(self, **kwargs)

    def get_url(self, path):
        return self._get_url(path).rstrip('/')

    @classmethod
    def from_parsedurl(cls, parsedurl, **kwargs):
        password = parsedurl.password
        if password is not None:
            kwargs['password'] = urlunquote(password)
        username = parsedurl.username
        if username is not None:
            kwargs['username'] = urlunquote(username)
        netloc = parsedurl.hostname
        if parsedurl.port:
            netloc = '%s:%s' % (netloc, parsedurl.port)
        if parsedurl.username:
            netloc = '%s@%s' % (parsedurl.username, netloc)
        parsedurl = parsedurl._replace(netloc=netloc)
        return cls(urlunparse(parsedurl), **kwargs)

    def __repr__(self):
        return '%s(%r, dumb=%r)' % (type(self).__name__, self._base_url, self.dumb)

    def _get_url(self, path):
        if not isinstance(path, str):
            path = path.decode('utf-8')
        return urljoin(self._base_url, path).rstrip('/') + '/'

    def _http_request(self, url, headers=None, data=None, allow_compression=False):
        """Perform HTTP request.

        Args:
          url: Request URL.
          headers: Optional custom headers to override defaults.
          data: Request data.
          allow_compression: Allow GZipped communication.

        Returns:
          Tuple (`response`, `read`), where response is an `urllib3`
          response object with additional `content_type` and
          `redirect_location` properties, and `read` is a consumable read
          method for the response data.

        """
        req_headers = self.pool_manager.headers.copy()
        if headers is not None:
            req_headers.update(headers)
        req_headers['Pragma'] = 'no-cache'
        if allow_compression:
            req_headers['Accept-Encoding'] = 'gzip'
        else:
            req_headers['Accept-Encoding'] = 'identity'
        if data is None:
            resp = self.pool_manager.request('GET', url, headers=req_headers)
        else:
            resp = self.pool_manager.request('POST', url, headers=req_headers, body=data)
        if resp.status == 404:
            raise NotGitRepository()
        if resp.status == 401:
            raise HTTPUnauthorized(resp.getheader('WWW-Authenticate'), url)
        if resp.status != 200:
            raise GitProtocolError('unexpected http resp %d for %s' % (resp.status, url))
        read = BytesIO(resp.data).read
        resp.content_type = resp.getheader('Content-Type')
        try:
            resp_url = resp.geturl()
        except AttributeError:
            resp.redirect_location = resp.get_redirect_location()
        else:
            resp.redirect_location = resp_url if resp_url != url else ''
        return (resp, read)

    def _discover_references(self, service, base_url):
        assert base_url[-1] == '/'
        tail = 'info/refs'
        headers = {'Accept': '*/*'}
        if self.dumb is not True:
            tail += '?service=%s' % service.decode('ascii')
        url = urljoin(base_url, tail)
        resp, read = self._http_request(url, headers, allow_compression=True)
        if resp.redirect_location:
            if not resp.redirect_location.endswith(tail):
                raise GitProtocolError('Redirected from URL %s to URL %s without %s' % (url, resp.redirect_location, tail))
            base_url = resp.redirect_location[:-len(tail)]
        try:
            self.dumb = not resp.content_type.startswith('application/x-git-')
            if not self.dumb:
                proto = Protocol(read, None)
                try:
                    [pkt] = list(proto.read_pkt_seq())
                except ValueError:
                    raise GitProtocolError('unexpected number of packets received')
                if pkt.rstrip(b'\n') != b'# service=' + service:
                    raise GitProtocolError('unexpected first line %r from smart server' % pkt)
                return read_pkt_refs(proto) + (base_url,)
            else:
                return (read_info_refs(resp), set(), base_url)
        finally:
            resp.close()

    def _smart_request(self, service, url, data):
        assert url[-1] == '/'
        url = urljoin(url, service)
        result_content_type = 'application/x-%s-result' % service
        headers = {'Content-Type': 'application/x-%s-request' % service, 'Accept': result_content_type, 'Content-Length': str(len(data))}
        resp, read = self._http_request(url, headers, data)
        if resp.content_type != result_content_type:
            raise GitProtocolError('Invalid content-type from server: %s' % resp.content_type)
        return (resp, read)

    def send_pack(self, path, update_refs, generate_pack_data, progress=None):
        """Upload a pack to a remote repository.

        Args:
          path: Repository path (as bytestring)
          update_refs: Function to determine changes to remote refs.
        Receives dict with existing remote refs, returns dict with
        changed refs (name -> sha, where sha=ZERO_SHA for deletions)
          generate_pack_data: Function that can return a tuple
        with number of elements and pack data to upload.
          progress: Optional progress function

        Returns:
          SendPackResult

        Raises:
          SendPackError: if server rejects the pack data

        """
        url = self._get_url(path)
        old_refs, server_capabilities, url = self._discover_references(b'git-receive-pack', url)
        negotiated_capabilities, agent = self._negotiate_receive_pack_capabilities(server_capabilities)
        negotiated_capabilities.add(capability_agent())
        if CAPABILITY_REPORT_STATUS in negotiated_capabilities:
            self._report_status_parser = ReportStatusParser()
        new_refs = update_refs(dict(old_refs))
        if new_refs is None:
            return SendPackResult(old_refs, agent=agent, ref_status={})
        if set(new_refs.items()).issubset(set(old_refs.items())):
            return SendPackResult(new_refs, agent=agent, ref_status={})
        if self.dumb:
            raise NotImplementedError(self.fetch_pack)
        req_data = BytesIO()
        req_proto = Protocol(None, req_data.write)
        have, want = self._handle_receive_pack_head(req_proto, negotiated_capabilities, old_refs, new_refs)
        pack_data_count, pack_data = generate_pack_data(have, want, ofs_delta=CAPABILITY_OFS_DELTA in negotiated_capabilities)
        if self._should_send_pack(new_refs):
            write_pack_data(req_proto.write_file(), pack_data_count, pack_data)
        resp, read = self._smart_request('git-receive-pack', url, data=req_data.getvalue())
        try:
            resp_proto = Protocol(read, None)
            ref_status = self._handle_receive_pack_tail(resp_proto, negotiated_capabilities, progress)
            return SendPackResult(new_refs, agent=agent, ref_status=ref_status)
        finally:
            resp.close()

    def fetch_pack(self, path, determine_wants, graph_walker, pack_data, progress=None, depth=None):
        """Retrieve a pack from a git smart server.

        Args:
          path: Path to fetch from
          determine_wants: Callback that returns list of commits to fetch
          graph_walker: Object with next() and ack().
          pack_data: Callback called for each bit of data in the pack
          progress: Callback for progress reports (strings)
          depth: Depth for request

        Returns:
          FetchPackResult object

        """
        url = self._get_url(path)
        refs, server_capabilities, url = self._discover_references(b'git-upload-pack', url)
        negotiated_capabilities, symrefs, agent = self._negotiate_upload_pack_capabilities(server_capabilities)
        wants = determine_wants(refs)
        if wants is not None:
            wants = [cid for cid in wants if cid != ZERO_SHA]
        if not wants:
            return FetchPackResult(refs, symrefs, agent)
        if self.dumb:
            raise NotImplementedError(self.fetch_pack)
        req_data = BytesIO()
        req_proto = Protocol(None, req_data.write)
        new_shallow, new_unshallow = self._handle_upload_pack_head(req_proto, negotiated_capabilities, graph_walker, wants, can_read=None, depth=depth)
        resp, read = self._smart_request('git-upload-pack', url, data=req_data.getvalue())
        try:
            resp_proto = Protocol(read, None)
            if new_shallow is None and new_unshallow is None:
                new_shallow, new_unshallow = _read_shallow_updates(resp_proto)
            self._handle_upload_pack_tail(resp_proto, negotiated_capabilities, graph_walker, pack_data, progress)
            return FetchPackResult(refs, symrefs, agent, new_shallow, new_unshallow)
        finally:
            resp.close()

    def get_refs(self, path):
        """Retrieve the current refs from a git smart server."""
        url = self._get_url(path)
        refs, _, _ = self._discover_references(b'git-upload-pack', url)
        return refs