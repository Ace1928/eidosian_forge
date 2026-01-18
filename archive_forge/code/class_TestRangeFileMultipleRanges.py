import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
class TestRangeFileMultipleRanges(tests.TestCase, TestRangeFileMixin):
    """Test a RangeFile for multiple ranges.

    The RangeFile used for the tests contains three ranges:

    - at offset 25: alpha
    - at offset 100: alpha
    - at offset 126: alpha.upper()

    The two last ranges are contiguous. This only rarely occurs (should not in
    fact) in real uses but may lead to hard to track bugs.
    """
    boundary = b'separation'

    def setUp(self):
        super().setUp()
        boundary = self.boundary
        content = b''
        self.first_range_start = 25
        file_size = 200
        for start, part in [(self.first_range_start, self.alpha), (100, self.alpha), (126, self.alpha.upper())]:
            content += self._multipart_byterange(part, start, boundary, file_size)
        content += self._boundary_line()
        self._file = response.RangeFile('Multiple_ranges_file', BytesIO(content))
        self.set_file_boundary()

    def _boundary_line(self):
        """Helper to build the formatted boundary line."""
        return b'--' + self.boundary + b'\r\n'

    def set_file_boundary(self):
        self._file.set_boundary(self.boundary)

    def _multipart_byterange(self, data, offset, boundary, file_size=b'*'):
        """Encode a part of a file as a multipart/byterange MIME type.

        When a range request is issued, the HTTP response body can be
        decomposed in parts, each one representing a range (start, size) in a
        file.

        :param data: The payload.
        :param offset: where data starts in the file
        :param boundary: used to separate the parts
        :param file_size: the size of the file containing the range (default to
            '*' meaning unknown)

        :return: a string containing the data encoded as it will appear in the
            HTTP response body.
        """
        bline = self._boundary_line()
        range = bline
        if isinstance(file_size, int):
            file_size = b'%d' % file_size
        range += b'Content-Range: bytes %d-%d/%s\r\n' % (offset, offset + len(data) - 1, file_size)
        range += b'\r\n'
        range += data
        return range

    def test_read_all_ranges(self):
        f = self._file
        self.assertEqual(self.alpha, f.read())
        f.seek(100)
        self.assertEqual(self.alpha, f.read())
        self.assertEqual(126, f.tell())
        f.seek(126)
        self.assertEqual(b'A', f.read(1))
        f.seek(10, 1)
        self.assertEqual(b'LMN', f.read(3))

    def test_seek_from_end(self):
        """See TestRangeFileMixin.test_seek_from_end."""
        f = self._file
        f.seek(-2, 2)
        self.assertEqual(b'yz', f.read())
        self.assertRaises(errors.InvalidRange, f.seek, -2, 2)

    def test_seek_into_void(self):
        f = self._file
        start = self.first_range_start
        f.seek(start)
        f.seek(start + 40)
        f.seek(100)
        f.seek(125)

    def test_seek_across_ranges(self):
        f = self._file
        f.seek(126)
        self.assertEqual(b'AB', f.read(2))

    def test_checked_read_dont_overflow_buffers(self):
        f = self._file
        f._discarded_buf_size = 8
        f.seek(126)
        self.assertEqual(b'AB', f.read(2))

    def test_seek_twice_between_ranges(self):
        f = self._file
        start = self.first_range_start
        f.seek(start + 40)
        self.assertRaises(errors.InvalidRange, f.seek, start + 41)

    def test_seek_at_range_end(self):
        """Test seek behavior at range end."""
        f = self._file
        f.seek(25 + 25)
        f.seek(100 + 25)
        f.seek(126 + 25)

    def test_read_at_range_end(self):
        f = self._file
        self.assertEqual(self.alpha, f.read())
        self.assertEqual(self.alpha, f.read())
        self.assertEqual(self.alpha.upper(), f.read())
        self.assertRaises(errors.InvalidHttpResponse, f.read, 1)