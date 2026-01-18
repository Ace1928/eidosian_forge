import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
class TestRangeFileMixin:
    """Tests for accessing the first range in a RangeFile."""
    alpha = b'abcdefghijklmnopqrstuvwxyz'

    def test_can_read_at_first_access(self):
        """Test that the just created file can be read."""
        self.assertEqual(self.alpha, self._file.read())

    def test_seek_read(self):
        """Test seek/read inside the range."""
        f = self._file
        start = self.first_range_start
        self.assertEqual(start, f.tell())
        cur = start
        f.seek(start + 3)
        cur += 3
        self.assertEqual(b'def', f.read(3))
        cur += len('def')
        f.seek(4, 1)
        cur += 4
        self.assertEqual(b'klmn', f.read(4))
        cur += len('klmn')
        self.assertEqual(b'', f.read(0))
        here = f.tell()
        f.seek(0, 1)
        self.assertEqual(here, f.tell())
        self.assertEqual(cur, f.tell())

    def test_read_zero(self):
        f = self._file
        self.assertEqual(b'', f.read(0))
        f.seek(10, 1)
        self.assertEqual(b'', f.read(0))

    def test_seek_at_range_end(self):
        f = self._file
        f.seek(26, 1)

    def test_read_at_range_end(self):
        """Test read behaviour at range end."""
        f = self._file
        self.assertEqual(self.alpha, f.read())
        self.assertEqual(b'', f.read(0))
        self.assertRaises(errors.InvalidRange, f.read, 1)

    def test_unbounded_read_after_seek(self):
        f = self._file
        f.seek(24, 1)
        self.assertEqual(b'yz', f.read())

    def test_seek_backwards(self):
        f = self._file
        start = self.first_range_start
        f.seek(start)
        f.read(12)
        self.assertRaises(errors.InvalidRange, f.seek, start + 5)

    def test_seek_outside_single_range(self):
        f = self._file
        if f._size == -1 or f._boundary is not None:
            raise tests.TestNotApplicable('Needs a fully defined range')
        self.assertRaises(errors.InvalidRange, f.seek, self.first_range_start + 27)

    def test_read_past_end_of_range(self):
        f = self._file
        if f._size == -1:
            raise tests.TestNotApplicable("Can't check an unknown size")
        start = self.first_range_start
        f.seek(start + 20)
        self.assertRaises(errors.InvalidRange, f.read, 10)

    def test_seek_from_end(self):
        """Test seeking from the end of the file.

        The semantic is unclear in case of multiple ranges. Seeking from end
        exists only for the http transports, cannot be used if the file size is
        unknown and is not used in breezy itself. This test must be (and is)
        overridden by daughter classes.

        Reading from end makes sense only when a range has been requested from
        the end of the file (see HttpTransportBase._get() when using the
        'tail_amount' parameter). The HTTP response can only be a whole file or
        a single range.
        """
        f = self._file
        f.seek(-2, 2)
        self.assertEqual(b'yz', f.read())