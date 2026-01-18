import base64
import cgi
import errno
import http.client
import os
import re
import socket
import ssl
import sys
import time
import urllib
import urllib.request
import weakref
from urllib.parse import urlencode, urljoin, urlparse
from ... import __version__ as breezy_version
from ... import config, debug, errors, osutils, trace, transport, ui, urlutils
from ...bzr.smart import medium
from ...trace import mutter, mutter_callsite
from ...transport import ConnectedTransport, NoSuchFile, UnusableRedirect
from . import default_user_agent, ssl
from .response import handle_response
class HttpTransport(ConnectedTransport):
    """HTTP Client implementations.

    The protocol can be given as e.g. http+urllib://host/ to use a particular
    implementation.
    """
    _debuglevel = 0

    def __init__(self, base, _from_transport=None, ca_certs=None):
        """Set the base path where files will be stored."""
        proto_match = re.match('^(https?)(\\+\\w+)?://', base)
        if not proto_match:
            raise AssertionError('not a http url: %r' % base)
        self._unqualified_scheme = proto_match.group(1)
        super().__init__(base, _from_transport=_from_transport)
        self._medium = None
        if _from_transport is not None:
            self._range_hint = _from_transport._range_hint
            self._opener = _from_transport._opener
        else:
            self._range_hint = 'multi'
            self._opener = Opener(report_activity=self._report_activity, ca_certs=ca_certs)

    def request(self, method, url, fields=None, headers=None, **urlopen_kw):
        body = urlopen_kw.pop('body', None)
        if fields is not None:
            data = urlencode(fields).encode()
            if body is not None:
                raise ValueError('body and fields are mutually exclusive')
        else:
            data = body
        if headers is None:
            headers = {}
        request = Request(method, url, data, headers)
        request.follow_redirections = urlopen_kw.pop('retries', 0) > 0
        if urlopen_kw:
            raise NotImplementedError('unknown arguments: %r' % urlopen_kw.keys())
        connection = self._get_connection()
        if connection is not None:
            request.connection = connection
            auth, proxy_auth = self._get_credentials()
            connection.cleanup_pipe()
        else:
            auth = self._create_auth()
            proxy_auth = dict()
        request.auth = auth
        request.proxy_auth = proxy_auth
        if self._debuglevel > 0:
            print('perform: {} base: {}, url: {}'.format(request.method, self.base, request.get_full_url()))
        response = self._opener.open(request)
        if self._get_connection() is not request.connection:
            self._set_connection(request.connection, (request.auth, request.proxy_auth))
        else:
            self._update_credentials((request.auth, request.proxy_auth))
        code = response.code
        if request.follow_redirections is False and code in (301, 302, 303, 307, 308):
            raise errors.RedirectRequested(request.get_full_url(), request.redirected_to, is_permanent=code in (301, 308))
        if request.redirected_to is not None:
            trace.mutter('redirected from: {} to: {}'.format(request.get_full_url(), request.redirected_to))

        class Urllib3LikeResponse:

            def __init__(self, actual):
                self._actual = actual
                self._data = None

            def getheader(self, name, default=None):
                if self._actual.headers is None:
                    raise http.client.ResponseNotReady()
                return self._actual.headers.get(name, default)

            def getheaders(self):
                if self._actual.headers is None:
                    raise http.client.ResponseNotReady()
                return list(self._actual.headers.items())

            @property
            def status(self):
                return self._actual.code

            @property
            def reason(self):
                return self._actual.reason

            @property
            def data(self):
                if self._data is None:
                    self._data = self._actual.read()
                return self._data

            @property
            def text(self):
                if self.status == 204:
                    return None
                charset = cgi.parse_header(self._actual.headers['Content-Type'])[1].get('charset')
                if charset:
                    return self.data.decode(charset)
                else:
                    return self.data.decode()

            def read(self, amt=None):
                if amt is None and 'evil' in debug.debug_flags:
                    mutter_callsite(4, 'reading full response.')
                return self._actual.read(amt)

            def readlines(self):
                return self._actual.readlines()

            def readline(self, size=-1):
                return self._actual.readline(size)
        return Urllib3LikeResponse(response)

    def disconnect(self):
        connection = self._get_connection()
        if connection is not None:
            connection.close()

    def has(self, relpath):
        """Does the target location exist?
        """
        response = self._head(relpath)
        code = response.status
        if code == 200:
            return True
        else:
            return False

    def get(self, relpath):
        """Get the file at the given relative path.

        :param relpath: The relative path to the file
        """
        code, response_file = self._get(relpath, None)
        return response_file

    def _get(self, relpath, offsets, tail_amount=0):
        """Get a file, or part of a file.

        :param relpath: Path relative to transport base URL
        :param offsets: None to get the whole file;
            or  a list of _CoalescedOffset to fetch parts of a file.
        :param tail_amount: The amount to get from the end of the file.

        :returns: (http_code, result_file)
        """
        abspath = self._remote_path(relpath)
        headers = {}
        if offsets or tail_amount:
            range_header = self._attempted_range_header(offsets, tail_amount)
            if range_header is not None:
                bytes = 'bytes=' + range_header
                headers = {'Range': bytes}
        else:
            range_header = None
        response = self.request('GET', abspath, headers=headers)
        if response.status == 404:
            raise NoSuchFile(abspath)
        elif response.status == 416:
            raise errors.InvalidHttpRange(abspath, range_header, 'Server return code %d' % response.status)
        elif response.status == 400:
            if range_header:
                raise errors.InvalidHttpRange(abspath, range_header, 'Server return code %d' % response.status)
            else:
                raise errors.BadHttpRequest(abspath, response.reason)
        elif response.status not in (200, 206):
            raise errors.UnexpectedHttpStatus(abspath, response.status, headers=response.getheaders())
        data = handle_response(abspath, response.status, response.getheader, response)
        return (response.status, data)

    def _remote_path(self, relpath):
        """See ConnectedTransport._remote_path.

        user and passwords are not embedded in the path provided to the server.
        """
        url = self._parsed_url.clone(relpath)
        url.user = url.quoted_user = None
        url.password = url.quoted_password = None
        url.scheme = self._unqualified_scheme
        return str(url)

    def _create_auth(self):
        """Returns a dict containing the credentials provided at build time."""
        auth = dict(host=self._parsed_url.host, port=self._parsed_url.port, user=self._parsed_url.user, password=self._parsed_url.password, protocol=self._unqualified_scheme, path=self._parsed_url.path)
        return auth

    def get_smart_medium(self):
        """See Transport.get_smart_medium."""
        if self._medium is None:
            self._medium = SmartClientHTTPMedium(self)
        return self._medium

    def _degrade_range_hint(self, relpath, ranges):
        if self._range_hint == 'multi':
            self._range_hint = 'single'
            mutter('Retry "%s" with single range request' % relpath)
        elif self._range_hint == 'single':
            self._range_hint = None
            mutter('Retry "%s" without ranges' % relpath)
        else:
            return False
        return True
    _bytes_to_read_before_seek = 128
    _max_readv_combine = 0
    _max_get_ranges = 200
    _get_max_size = 0

    def _readv(self, relpath, offsets):
        """Get parts of the file at the given relative path.

        :param offsets: A list of (offset, size) tuples.
        :param return: A list or generator of (offset, data) tuples
        """
        offsets = list(offsets)
        try_again = True
        retried_offset = None
        while try_again:
            try_again = False
            sorted_offsets = sorted(offsets)
            coalesced = self._coalesce_offsets(sorted_offsets, limit=self._max_readv_combine, fudge_factor=self._bytes_to_read_before_seek, max_size=self._get_max_size)
            coalesced = list(coalesced)
            if 'http' in debug.debug_flags:
                mutter('http readv of %s  offsets => %s collapsed %s', relpath, len(offsets), len(coalesced))
            data_map = {}
            iter_offsets = iter(offsets)
            try:
                cur_offset_and_size = next(iter_offsets)
            except StopIteration:
                return
            try:
                for cur_coal, rfile in self._coalesce_readv(relpath, coalesced):
                    for offset, size in cur_coal.ranges:
                        start = cur_coal.start + offset
                        rfile.seek(start, os.SEEK_SET)
                        data = rfile.read(size)
                        data_len = len(data)
                        if data_len != size:
                            raise errors.ShortReadvError(relpath, start, size, actual=data_len)
                        if (start, size) == cur_offset_and_size:
                            yield (cur_offset_and_size[0], data)
                            try:
                                cur_offset_and_size = next(iter_offsets)
                            except StopIteration:
                                return
                        else:
                            data_map[start, size] = data
                    while cur_offset_and_size in data_map:
                        this_data = data_map.pop(cur_offset_and_size)
                        yield (cur_offset_and_size[0], this_data)
                        try:
                            cur_offset_and_size = next(iter_offsets)
                        except StopIteration:
                            return
            except (errors.ShortReadvError, errors.InvalidRange, errors.InvalidHttpRange, errors.HttpBoundaryMissing) as e:
                mutter('Exception %r: %s during http._readv', e, e)
                if not isinstance(e, errors.ShortReadvError) or retried_offset == cur_offset_and_size:
                    if not self._degrade_range_hint(relpath, coalesced):
                        raise
                offsets = [cur_offset_and_size] + [o for o in iter_offsets]
                retried_offset = cur_offset_and_size
                try_again = True

    def _coalesce_readv(self, relpath, coalesced):
        """Issue several GET requests to satisfy the coalesced offsets"""

        def get_and_yield(relpath, coalesced):
            if coalesced:
                code, rfile = self._get(relpath, coalesced)
                for coal in coalesced:
                    yield (coal, rfile)
        if self._range_hint is None:
            yield from get_and_yield(relpath, coalesced)
        else:
            total = len(coalesced)
            if self._range_hint == 'multi':
                max_ranges = self._max_get_ranges
            elif self._range_hint == 'single':
                max_ranges = total
            else:
                raise AssertionError('Unknown _range_hint %r' % (self._range_hint,))
            cumul = 0
            ranges = []
            for coal in coalesced:
                if self._get_max_size > 0 and cumul + coal.length > self._get_max_size or len(ranges) >= max_ranges:
                    yield from get_and_yield(relpath, ranges)
                    ranges = [coal]
                    cumul = coal.length
                else:
                    ranges.append(coal)
                    cumul += coal.length
            yield from get_and_yield(relpath, ranges)

    def recommended_page_size(self):
        """See Transport.recommended_page_size().

        For HTTP we suggest a large page size to reduce the overhead
        introduced by latency.
        """
        return 64 * 1024

    def _post(self, body_bytes):
        """POST body_bytes to .bzr/smart on this transport.

        :returns: (response code, response body file-like object).
        """
        abspath = self._remote_path('.bzr/smart')
        response = self.request('POST', abspath, body=body_bytes, headers={'Content-Type': 'application/octet-stream'})
        code = response.status
        data = handle_response(abspath, code, response.getheader, response)
        return (code, data)

    def _head(self, relpath):
        """Request the HEAD of a file.

        Performs the request and leaves callers handle the results.
        """
        abspath = self._remote_path(relpath)
        response = self.request('HEAD', abspath)
        if response.status not in (200, 404):
            raise errors.UnexpectedHttpStatus(abspath, response.status, headers=response.getheaders())
        return response
        raise NotImplementedError(self._post)

    def put_file(self, relpath, f, mode=None):
        """Copy the file-like object into the location.

        :param relpath: Location to put the contents, relative to base.
        :param f:       File-like object.
        """
        raise errors.TransportNotPossible('http PUT not supported')

    def mkdir(self, relpath, mode=None):
        """Create a directory at the given path."""
        raise errors.TransportNotPossible('http does not support mkdir()')

    def rmdir(self, relpath):
        """See Transport.rmdir."""
        raise errors.TransportNotPossible('http does not support rmdir()')

    def append_file(self, relpath, f, mode=None):
        """Append the text in the file-like object into the final
        location.
        """
        raise errors.TransportNotPossible('http does not support append()')

    def copy(self, rel_from, rel_to):
        """Copy the item at rel_from to the location at rel_to"""
        raise errors.TransportNotPossible('http does not support copy()')

    def copy_to(self, relpaths, other, mode=None, pb=None):
        """Copy a set of entries from self into another Transport.

        :param relpaths: A list/generator of entries to be copied.

        TODO: if other is LocalTransport, is it possible to
              do better than put(get())?
        """
        if isinstance(other, HttpTransport):
            raise errors.TransportNotPossible('http cannot be the target of copy_to()')
        else:
            return super().copy_to(relpaths, other, mode=mode, pb=pb)

    def move(self, rel_from, rel_to):
        """Move the item at rel_from to the location at rel_to"""
        raise errors.TransportNotPossible('http does not support move()')

    def delete(self, relpath):
        """Delete the item at relpath"""
        raise errors.TransportNotPossible('http does not support delete()')

    def external_url(self):
        """See breezy.transport.Transport.external_url."""
        url = self._parsed_url.clone()
        url.scheme = self._unqualified_scheme
        return str(url)

    def is_readonly(self):
        """See Transport.is_readonly."""
        return True

    def listable(self):
        """See Transport.listable."""
        return False

    def stat(self, relpath):
        """Return the stat information for a file.
        """
        raise errors.TransportNotPossible('http does not support stat()')

    def lock_read(self, relpath):
        """Lock the given file for shared (read) access.
        :return: A lock object, which should be passed to Transport.unlock()
        """

        class BogusLock:

            def __init__(self, path):
                self.path = path

            def unlock(self):
                pass
        return BogusLock(relpath)

    def lock_write(self, relpath):
        """Lock the given file for exclusive (write) access.
        WARNING: many transports do not support this, so trying avoid using it

        :return: A lock object, which should be passed to Transport.unlock()
        """
        raise errors.TransportNotPossible('http does not support lock_write()')

    def _attempted_range_header(self, offsets, tail_amount):
        """Prepare a HTTP Range header at a level the server should accept.

        :return: the range header representing offsets/tail_amount or None if
            no header can be built.
        """
        if self._range_hint == 'multi':
            return self._range_header(offsets, tail_amount)
        elif self._range_hint == 'single':
            if len(offsets) > 0:
                if tail_amount not in (0, None):
                    return None
                else:
                    start = offsets[0].start
                    last = offsets[-1]
                    end = last.start + last.length - 1
                    whole = self._coalesce_offsets([(start, end - start + 1)], limit=0, fudge_factor=0)
                    return self._range_header(list(whole), 0)
            else:
                return self._range_header(offsets, tail_amount)
        else:
            return None

    @staticmethod
    def _range_header(ranges, tail_amount):
        """Turn a list of bytes ranges into a HTTP Range header value.

        :param ranges: A list of _CoalescedOffset
        :param tail_amount: The amount to get from the end of the file.

        :return: HTTP range header string.

        At least a non-empty ranges *or* a tail_amount must be
        provided.
        """
        strings = []
        for offset in ranges:
            strings.append('%d-%d' % (offset.start, offset.start + offset.length - 1))
        if tail_amount:
            strings.append('-%d' % tail_amount)
        return ','.join(strings)

    def _redirected_to(self, source, target):
        """Returns a transport suitable to re-issue a redirected request.

        :param source: The source url as returned by the server.
        :param target: The target url as returned by the server.

        The redirection can be handled only if the relpath involved is not
        renamed by the redirection.

        :returns: A transport
        :raise UnusableRedirect: when the URL can not be reinterpreted
        """
        parsed_source = self._split_url(source)
        parsed_target = self._split_url(target)
        pl = len(self._parsed_url.path)
        excess_tail = parsed_source.path[pl:].strip('/')
        if not parsed_target.path.endswith(excess_tail):
            raise UnusableRedirect(source, target, 'final part of the url was renamed')
        target_path = parsed_target.path
        if excess_tail:
            target_path = target_path[:-len(excess_tail)]
        if parsed_target.scheme in ('http', 'https'):
            if parsed_target.scheme == self._unqualified_scheme and parsed_target.host == self._parsed_url.host and (parsed_target.port == self._parsed_url.port) and (parsed_target.user is None or parsed_target.user == self._parsed_url.user):
                return self.clone(target_path)
            else:
                redir_scheme = parsed_target.scheme
                new_url = self._unsplit_url(redir_scheme, self._parsed_url.user, self._parsed_url.password, parsed_target.host, parsed_target.port, target_path)
                return transport.get_transport_from_url(new_url)
        else:
            new_url = self._unsplit_url(parsed_target.scheme, parsed_target.user, parsed_target.password, parsed_target.host, parsed_target.port, target_path)
            return transport.get_transport_from_url(new_url)

    def _options(self, relpath):
        abspath = self._remote_path(relpath)
        resp = self.request('OPTIONS', abspath)
        if resp.status == 404:
            raise NoSuchFile(abspath)
        if resp.status in (403, 405):
            raise errors.InvalidHttpResponse(abspath, 'OPTIONS not supported or forbidden for remote URL', headers=resp.getheaders())
        return resp.getheaders()