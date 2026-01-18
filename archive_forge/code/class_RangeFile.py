import cgi
import email.utils as email_utils
import http.client as http_client
import os
from io import BytesIO
from ... import errors, osutils
class RangeFile(ResponseFile):
    """File-like object that allow access to partial available data.

    All accesses should happen sequentially since the acquisition occurs during
    an http response reception (as sockets can't be seeked, we simulate the
    seek by just reading and discarding the data).

    The access pattern is defined by a set of ranges discovered as reading
    progress. Only one range is available at a given time, so all accesses
    should happen with monotonically increasing offsets.
    """
    _discarded_buf_size = 8192
    _max_read_size = 512 * 1024

    def __init__(self, path, infile):
        """Constructor.

        :param path: File url, for error reports.

        :param infile: File-like socket set at body start.
        """
        super().__init__(path, infile)
        self._boundary = None
        self._headers = None
        self.set_range(0, -1)

    def set_range(self, start, size):
        """Change the range mapping"""
        self._start = start
        self._size = size
        self._pos = self._start

    def set_boundary(self, boundary):
        """Define the boundary used in a multi parts message.

        The file should be at the beginning of the body, the first range
        definition is read and taken into account.
        """
        if not isinstance(boundary, bytes):
            raise TypeError(boundary)
        self._boundary = boundary
        self.read_boundary()
        self.read_range_definition()

    def read_boundary(self):
        """Read the boundary headers defining a new range"""
        boundary_line = b'\r\n'
        while boundary_line == b'\r\n':
            boundary_line = self._file.readline()
        if boundary_line == b'':
            raise errors.HttpBoundaryMissing(self._path, self._boundary)
        if boundary_line != b'--' + self._boundary + b'\r\n':
            if self._unquote_boundary(boundary_line) != b'--' + self._boundary + b'\r\n':
                raise errors.InvalidHttpResponse(self._path, "Expected a boundary (%s) line, got '%s'" % (self._boundary, boundary_line))

    def _unquote_boundary(self, b):
        return b[:2] + email_utils.unquote(b[2:-2].decode('ascii')).encode('ascii') + b[-2:]

    def read_range_definition(self):
        """Read a new range definition in a multi parts message.

        Parse the headers including the empty line following them so that we
        are ready to read the data itself.
        """
        self._headers = http_client.parse_headers(self._file)
        content_range = self._headers.get('content-range', None)
        if content_range is None:
            raise errors.InvalidHttpResponse(self._path, 'Content-Range header missing in a multi-part response', headers=self._headers)
        self.set_range_from_header(content_range)

    def set_range_from_header(self, content_range):
        """Helper to set the new range from its description in the headers"""
        try:
            rtype, values = content_range.split()
        except ValueError:
            raise errors.InvalidHttpRange(self._path, content_range, 'Malformed header')
        if rtype != 'bytes':
            raise errors.InvalidHttpRange(self._path, content_range, "Unsupported range type '%s'" % rtype)
        try:
            start_end, total = values.split('/')
            start, end = start_end.split('-')
            start = int(start)
            end = int(end)
        except ValueError:
            raise errors.InvalidHttpRange(self._path, content_range, 'Invalid range values')
        size = end - start + 1
        if size <= 0:
            raise errors.InvalidHttpRange(self._path, content_range, 'Invalid range, size <= 0')
        self.set_range(start, size)

    def _checked_read(self, size):
        """Read the file checking for short reads.

        The data read is discarded along the way.
        """
        pos = self._pos
        remaining = size
        while remaining > 0:
            data = self._file.read(min(remaining, self._discarded_buf_size))
            remaining -= len(data)
            if not data:
                raise errors.ShortReadvError(self._path, pos, size, size - remaining)
        self._pos += size

    def _seek_to_next_range(self):
        if self._boundary is None:
            raise errors.InvalidRange(self._path, self._pos, 'Range (%s, %s) exhausted' % (self._start, self._size))
        self.read_boundary()
        self.read_range_definition()

    def read(self, size=-1):
        """Read size bytes from the current position in the file.

        Reading across ranges is not supported. We rely on the underlying http
        client to clean the socket if we leave bytes unread. This may occur for
        the final boundary line of a multipart response or for any range
        request not entirely consumed by the client (due to offset coalescing)

        :param size:  The number of bytes to read.  Leave unspecified or pass
            -1 to read to EOF.
        """
        if self._size > 0 and self._pos == self._start + self._size:
            if size == 0:
                return b''
            else:
                self._seek_to_next_range()
        elif self._pos < self._start:
            raise errors.InvalidRange(self._path, self._pos, "Can't read %s bytes before range (%s, %s)" % (size, self._start, self._size))
        if self._size > 0:
            if size > 0 and self._pos + size > self._start + self._size:
                raise errors.InvalidRange(self._path, self._pos, "Can't read %s bytes across range (%s, %s)" % (size, self._start, self._size))
        buf = BytesIO()
        limited = size
        if self._size > 0:
            limited = self._start + self._size - self._pos
            if size >= 0:
                limited = min(limited, size)
        osutils.pumpfile(self._file, buf, limited, self._max_read_size)
        data = buf.getvalue()
        self._pos += len(data)
        return data

    def seek(self, offset, whence=0):
        start_pos = self._pos
        if whence == 0:
            final_pos = offset
        elif whence == 1:
            final_pos = start_pos + offset
        elif whence == 2:
            if self._size > 0:
                final_pos = self._start + self._size + offset
            else:
                raise errors.InvalidRange(self._path, self._pos, "RangeFile: can't seek from end while size is unknown")
        else:
            raise ValueError('Invalid value %s for whence.' % whence)
        if final_pos < self._pos:
            raise errors.InvalidRange(self._path, self._pos, 'RangeFile: trying to seek backwards to %s' % final_pos)
        if self._size > 0:
            cur_limit = self._start + self._size
            while final_pos > cur_limit:
                remain = cur_limit - self._pos
                if remain > 0:
                    self._checked_read(remain)
                self._seek_to_next_range()
                cur_limit = self._start + self._size
        size = final_pos - self._pos
        if size > 0:
            self._checked_read(size)

    def tell(self):
        return self._pos