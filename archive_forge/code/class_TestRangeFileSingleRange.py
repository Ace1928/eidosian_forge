import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
class TestRangeFileSingleRange(tests.TestCase, TestRangeFileMixin):
    """Test a RangeFile for a single range."""

    def setUp(self):
        super().setUp()
        self._file = response.RangeFile('Single_range_file', BytesIO(self.alpha))
        self.first_range_start = 15
        self._file.set_range(self.first_range_start, len(self.alpha))

    def test_read_before_range(self):
        f = self._file
        f._pos = 0
        self.assertRaises(errors.InvalidRange, f.read, 2)