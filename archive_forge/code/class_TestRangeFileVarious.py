import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
class TestRangeFileVarious(tests.TestCase):
    """Tests RangeFile aspects not covered elsewhere."""

    def test_seek_whence(self):
        """Test the seek whence parameter values."""
        f = response.RangeFile('foo', BytesIO(b'abc'))
        f.set_range(0, 3)
        f.seek(0)
        f.seek(1, 1)
        f.seek(-1, 2)
        self.assertRaises(ValueError, f.seek, 0, 14)

    def test_range_syntax(self):
        """Test the Content-Range scanning."""
        f = response.RangeFile('foo', BytesIO())

        def ok(expected, header_value):
            f.set_range_from_header(header_value)
            self.assertEqual(expected, (f.tell(), f._size))
        ok((1, 10), 'bytes 1-10/11')
        ok((1, 10), 'bytes 1-10/*')
        ok((12, 2), '\tbytes 12-13/*')
        ok((28, 1), '  bytes 28-28/*')
        ok((2123, 2120), 'bytes  2123-4242/12310')
        ok((1, 10), 'bytes 1-10/ttt')

        def nok(header_value):
            self.assertRaises(errors.InvalidHttpRange, f.set_range_from_header, header_value)
        nok('bytes 10-2/3')
        nok('chars 1-2/3')
        nok('bytes xx-yyy/zzz')
        nok('bytes xx-12/zzz')
        nok('bytes 11-yy/zzz')
        nok('bytes10-2/3')