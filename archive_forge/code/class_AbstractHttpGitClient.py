import logging
import os
import select
import socket
import subprocess
import sys
from contextlib import closing
from io import BufferedReader, BytesIO
from typing import (
from urllib.parse import quote as urlquote
from urllib.parse import unquote as urlunquote
from urllib.parse import urljoin, urlparse, urlunparse, urlunsplit
import dulwich
from .config import Config, apply_instead_of, get_xdg_config_home_path
from .errors import GitProtocolError, NotGitRepository, SendPackError
from .pack import (
from .protocol import (
from .refs import PEELED_TAG_SUFFIX, _import_remote_refs, read_info_refs
from .repo import Repo
class AbstractHttpGitClient(GitClient):
    """Abstract base class for HTTP Git Clients.

    This is agonistic of the actual HTTP implementation.

    Subclasses should provide an implementation of the
    _http_request method.
    """

    def __init__(self, base_url, dumb=False, **kwargs) -> None:
        self._base_url = base_url.rstrip('/') + '/'
        self.dumb = dumb
        GitClient.__init__(self, **kwargs)

    def _http_request(self, url, headers=None, data=None):
        """Perform HTTP request.

        Args:
          url: Request URL.
          headers: Optional custom headers to override defaults.
          data: Request data.

        Returns:
          Tuple (response, read), where response is an urllib3
          response object with additional content_type and
          redirect_location properties, and read is a consumable read
          method for the response data.

        Raises:
          GitProtocolError
        """
        raise NotImplementedError(self._http_request)

    def _discover_references(self, service, base_url):
        assert base_url[-1] == '/'
        tail = 'info/refs'
        headers = {'Accept': '*/*'}
        if self.dumb is not True:
            tail += '?service=%s' % service.decode('ascii')
        url = urljoin(base_url, tail)
        resp, read = self._http_request(url, headers)
        if resp.redirect_location:
            if not resp.redirect_location.endswith(tail):
                raise GitProtocolError(f'Redirected from URL {url} to URL {resp.redirect_location} without {tail}')
            base_url = urljoin(url, resp.redirect_location[:-len(tail)])
        try:
            self.dumb = resp.content_type is None or not resp.content_type.startswith('application/x-git-')
            if not self.dumb:
                proto = Protocol(read, None)
                try:
                    [pkt] = list(proto.read_pkt_seq())
                except ValueError as exc:
                    raise GitProtocolError('unexpected number of packets received') from exc
                if pkt.rstrip(b'\n') != b'# service=' + service:
                    raise GitProtocolError('unexpected first line %r from smart server' % pkt)
                return (*read_pkt_refs(proto.read_pkt_seq()), base_url)
            else:
                return (read_info_refs(resp), set(), base_url)
        finally:
            resp.close()

    def _smart_request(self, service, url, data):
        """Send a 'smart' HTTP request.

        This is a simple wrapper around _http_request that sets
        a couple of extra headers.
        """
        assert url[-1] == '/'
        url = urljoin(url, service)
        result_content_type = 'application/x-%s-result' % service
        headers = {'Content-Type': 'application/x-%s-request' % service, 'Accept': result_content_type}
        if isinstance(data, bytes):
            headers['Content-Length'] = str(len(data))
        resp, read = self._http_request(url, headers, data)
        if resp.content_type.split(';')[0] != result_content_type:
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

        def body_generator():
            header_handler = _v1ReceivePackHeader(negotiated_capabilities, old_refs, new_refs)
            for pkt in header_handler:
                yield pkt_line(pkt)
            pack_data_count, pack_data = generate_pack_data(header_handler.have, header_handler.want, ofs_delta=CAPABILITY_OFS_DELTA in negotiated_capabilities)
            if self._should_send_pack(new_refs):
                yield from PackChunkGenerator(pack_data_count, pack_data)
        resp, read = self._smart_request('git-receive-pack', url, data=body_generator())
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
        if depth is not None:
            wants = determine_wants(refs, depth=depth)
        else:
            wants = determine_wants(refs)
        if wants is not None:
            wants = [cid for cid in wants if cid != ZERO_SHA]
        if not wants:
            return FetchPackResult(refs, symrefs, agent)
        if self.dumb:
            raise NotImplementedError(self.fetch_pack)
        req_data = BytesIO()
        req_proto = Protocol(None, req_data.write)
        new_shallow, new_unshallow = _handle_upload_pack_head(req_proto, negotiated_capabilities, graph_walker, wants, can_read=None, depth=depth)
        resp, read = self._smart_request('git-upload-pack', url, data=req_data.getvalue())
        try:
            resp_proto = Protocol(read, None)
            if new_shallow is None and new_unshallow is None:
                new_shallow, new_unshallow = _read_shallow_updates(resp_proto.read_pkt_seq())
            _handle_upload_pack_tail(resp_proto, negotiated_capabilities, graph_walker, pack_data, progress)
            return FetchPackResult(refs, symrefs, agent, new_shallow, new_unshallow)
        finally:
            resp.close()

    def get_refs(self, path):
        """Retrieve the current refs from a git smart server."""
        url = self._get_url(path)
        refs, _, _ = self._discover_references(b'git-upload-pack', url)
        return refs

    def get_url(self, path):
        return self._get_url(path).rstrip('/')

    def _get_url(self, path):
        return urljoin(self._base_url, path).rstrip('/') + '/'

    @classmethod
    def from_parsedurl(cls, parsedurl, **kwargs):
        password = parsedurl.password
        if password is not None:
            kwargs['password'] = urlunquote(password)
        username = parsedurl.username
        if username is not None:
            kwargs['username'] = urlunquote(username)
        return cls(urlunparse(parsedurl), **kwargs)

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self._base_url!r}, dumb={self.dumb!r})'