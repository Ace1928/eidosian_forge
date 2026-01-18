import errno
import http.client as http_client
import http.server as http_server
import os
import posixpath
import random
import re
import socket
import sys
from urllib.parse import urlparse
from .. import osutils, urlutils
from . import test_server
class TestingHTTPRequestHandler(http_server.SimpleHTTPRequestHandler):
    """Handles one request.

    A TestingHTTPRequestHandler is instantiated for every request received by
    the associated server. Note that 'request' here is inherited from the base
    TCPServer class, for the HTTP server it is really a connection which itself
    will handle one or several HTTP requests.
    """
    protocol_version = 'HTTP/1.1'
    MessageClass = http_client.HTTPMessage

    def setup(self):
        http_server.SimpleHTTPRequestHandler.setup(self)
        self._cwd = self.server._home_dir
        tcs = self.server.test_case_server
        if tcs.protocol_version is not None:
            self.protocol_version = tcs.protocol_version

    def log_message(self, format, *args):
        tcs = self.server.test_case_server
        tcs.log('webserver - %s - - [%s] %s "%s" "%s"', self.address_string(), self.log_date_time_string(), format % args, self.headers.get('referer', '-'), self.headers.get('user-agent', '-'))

    def handle_one_request(self):
        """Handle a single HTTP request.

        We catch all socket errors occurring when the client close the
        connection early to avoid polluting the test results.
        """
        try:
            self._handle_one_request()
        except OSError as e:
            self.close_connection = 1
            if len(e.args) == 0 or e.args[0] not in (errno.EPIPE, errno.ECONNRESET, errno.ECONNABORTED, errno.EBADF):
                raise
    error_content_type = 'text/plain'
    error_message_format = 'Error code: %(code)s.\nMessage: %(message)s.\n'

    def send_error(self, code, message=None):
        """Send and log an error reply.

        We redefine the python-provided version to be able to set a
        ``Content-Length`` header as some http/1.1 clients complain otherwise
        (see bug #568421).

        :param code: The HTTP error code.

        :param message: The explanation of the error code, Defaults to a short
             entry.
        """
        if message is None:
            try:
                message = self.responses[code][0]
            except KeyError:
                message = '???'
        self.log_error('code %d, message %s', code, message)
        content = self.error_message_format % {'code': code, 'message': message}
        self.send_response(code, message)
        self.send_header('Content-Type', self.error_content_type)
        self.send_header('Content-Length', '%d' % len(content))
        self.send_header('Connection', 'close')
        self.end_headers()
        if self.command != 'HEAD' and code >= 200 and (code not in (204, 304)):
            self.wfile.write(content.encode('utf-8'))

    def _handle_one_request(self):
        http_server.SimpleHTTPRequestHandler.handle_one_request(self)
    _range_regexp = re.compile('^(?P<start>\\d+)-(?P<end>\\d+)?$')
    _tail_regexp = re.compile('^-(?P<tail>\\d+)$')

    def _parse_ranges(self, ranges_header, file_size):
        """Parse the range header value and returns ranges.

        RFC2616 14.35 says that syntactically invalid range specifiers MUST be
        ignored. In that case, we return None instead of a range list.

        :param ranges_header: The 'Range' header value.

        :param file_size: The size of the requested file.

        :return: A list of (start, end) tuples or None if some invalid range
            specifier is encountered.
        """
        if not ranges_header.startswith('bytes='):
            return None
        tail = None
        ranges = []
        ranges_header = ranges_header[len('bytes='):]
        for range_str in ranges_header.split(','):
            range_match = self._range_regexp.match(range_str)
            if range_match is not None:
                start = int(range_match.group('start'))
                end_match = range_match.group('end')
                if end_match is None:
                    end = file_size
                else:
                    end = int(end_match)
                if start > end:
                    return None
                ranges.append((start, end))
            else:
                tail_match = self._tail_regexp.match(range_str)
                if tail_match is not None:
                    tail = int(tail_match.group('tail'))
                else:
                    return None
        if tail is not None:
            ranges.append((max(0, file_size - tail), file_size))
        checked_ranges = []
        for start, end in ranges:
            if start >= file_size:
                return None
            end = min(end, file_size - 1)
            checked_ranges.append((start, end))
        return checked_ranges

    def _header_line_length(self, keyword, value):
        header_line = '{}: {}\r\n'.format(keyword, value)
        return len(header_line)

    def send_range_content(self, file, start, length):
        file.seek(start)
        self.wfile.write(file.read(length))

    def get_single_range(self, file, file_size, start, end):
        self.send_response(206)
        length = end - start + 1
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', '%d' % length)
        self.send_header('Content-Type', 'application/octet-stream')
        self.send_header('Content-Range', 'bytes %d-%d/%d' % (start, end, file_size))
        self.end_headers()
        self.send_range_content(file, start, length)

    def get_multiple_ranges(self, file, file_size, ranges):
        self.send_response(206)
        self.send_header('Accept-Ranges', 'bytes')
        boundary = '%d' % random.randint(0, 2147483647)
        self.send_header('Content-Type', 'multipart/byteranges; boundary=%s' % boundary)
        boundary_line = b'--%s\r\n' % boundary.encode('ascii')
        content_length = 0
        for start, end in ranges:
            content_length += len(boundary_line)
            content_length += self._header_line_length('Content-type', 'application/octet-stream')
            content_length += self._header_line_length('Content-Range', 'bytes %d-%d/%d' % (start, end, file_size))
            content_length += len('\r\n')
            content_length += end - start + 1
        content_length += len(boundary_line)
        self.send_header('Content-length', content_length)
        self.end_headers()
        for start, end in ranges:
            self.wfile.write(boundary_line)
            self.send_header('Content-type', 'application/octet-stream')
            self.send_header('Content-Range', 'bytes %d-%d/%d' % (start, end, file_size))
            self.end_headers()
            self.send_range_content(file, start, end - start + 1)
        self.wfile.write(boundary_line)

    def do_GET(self):
        """Serve a GET request.

        Handles the Range header.
        """
        self.server.test_case_server.GET_request_nb += 1
        path = self.translate_path(self.path)
        ranges_header_value = self.headers.get('Range')
        if ranges_header_value is None or os.path.isdir(path):
            return http_server.SimpleHTTPRequestHandler.do_GET(self)
        try:
            f = open(path, 'rb')
        except OSError:
            self.send_error(404, 'File not found')
            return
        file_size = os.fstat(f.fileno())[6]
        ranges = self._parse_ranges(ranges_header_value, file_size)
        if not ranges:
            f.close()
            self.send_error(416, 'Requested range not satisfiable')
            return
        if len(ranges) == 1:
            start, end = ranges[0]
            self.get_single_range(f, file_size, start, end)
        else:
            self.get_multiple_ranges(f, file_size, ranges)
        f.close()

    def translate_path(self, path):
        """Translate a /-separated PATH to the local filename syntax.

        If the server requires it, proxy the path before the usual translation
        """
        if self.server.test_case_server.proxy_requests:
            path = urlparse(path)[2]
            path += '-proxied'
        return self._translate_path(path)

    def _translate_path(self, path):
        """Translate a /-separated PATH to the local filename syntax.

        Note that we're translating http URLs here, not file URLs.
        The URL root location is the server's startup directory.
        Components that mean special things to the local file system
        (e.g. drive or directory names) are ignored.  (XXX They should
        probably be diagnosed.)

        Override from python standard library to stop it calling os.getcwd()
        """
        path = urlparse(path)[2]
        path = posixpath.normpath(urlutils.unquote(path))
        words = path.split('/')
        path = self._cwd
        for num, word in enumerate((w for w in words if w)):
            if num == 0:
                drive, word = os.path.splitdrive(word)
            head, word = os.path.split(word)
            if word in (os.curdir, os.pardir):
                continue
            path = os.path.join(path, word)
        return path