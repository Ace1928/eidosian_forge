import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
class TestRangeFileSizeUnknown(tests.TestCase, TestRangeFileMixin):
    """Test a RangeFile for a whole file whose size is not known."""

    def setUp(self):
        super().setUp()
        self._file = response.RangeFile('Whole_file_size_known', BytesIO(self.alpha))
        self.first_range_start = 0

    def test_seek_from_end(self):
        """See TestRangeFileMixin.test_seek_from_end.

        The end of the file can't be determined since the size is unknown.
        """
        self.assertRaises(errors.InvalidRange, self._file.seek, -1, 2)

    def test_read_at_range_end(self):
        """Test read behaviour at range end."""
        f = self._file
        self.assertEqual(self.alpha, f.read())
        self.assertEqual(b'', f.read(0))
        self.assertEqual(b'', f.read(1))