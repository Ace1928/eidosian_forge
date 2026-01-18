from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
from gslib.file_part import FilePart
import gslib.tests.testcase as testcase
class TestFilePart(testcase.GsUtilUnitTestCase):
    """Unit tests for FilePart class."""

    def test_tell(self):
        filename = 'test_tell'
        contents = 100 * b'x'
        fpath = self.CreateTempFile(file_name=filename, contents=contents)
        part_length = 23
        start_pos = 50
        fp = FilePart(fpath, start_pos, part_length)
        self.assertEqual(start_pos, fp._fp.tell())
        self.assertEqual(0, fp.tell())

    def test_seek(self):
        """Tests seeking in a FilePart."""
        filename = 'test_seek'
        contents = 100 * b'x'
        part_length = 23
        start_pos = 50
        fpath = self.CreateTempFile(file_name=filename, contents=contents)
        fp = FilePart(fpath, start_pos, part_length)
        offset = 10
        fp.seek(offset)
        self.assertEqual(start_pos + offset, fp._fp.tell())
        self.assertEqual(offset, fp.tell())
        fp.seek(offset, whence=os.SEEK_CUR)
        self.assertEqual(start_pos + 2 * offset, fp._fp.tell())
        self.assertEqual(2 * offset, fp.tell())
        fp.seek(-offset, whence=os.SEEK_END)
        self.assertEqual(start_pos + part_length - offset, fp._fp.tell())
        self.assertEqual(part_length - offset, fp.tell())
        fp.seek(1, whence=os.SEEK_END)
        self.assertEqual(start_pos + part_length + 1, fp._fp.tell())
        self.assertEqual(part_length + 1, fp.tell())

    def test_read(self):
        """Tests various reaad operations with FilePart."""
        filename = 'test_read'
        contents = bytearray(range(256))
        part_length = 23
        start_pos = 50
        fpath = self.CreateTempFile(file_name=filename, contents=contents)
        fp = FilePart(fpath, start_pos, part_length)
        whole_file = fp.read()
        self.assertEqual(contents[start_pos:start_pos + part_length], whole_file)
        fp.seek(0)
        offset = 10
        partial_file = fp.read(offset)
        self.assertEqual(contents[start_pos:start_pos + offset], partial_file)
        remaining_file = fp.read(part_length - offset)
        self.assertEqual(contents[start_pos + offset:start_pos + part_length], remaining_file)
        self.assertEqual(contents[start_pos:start_pos + part_length], partial_file + remaining_file)
        empty_file = fp.read(100)
        self.assertEqual(b'', empty_file)
        empty_file = fp.read()
        self.assertEqual(b'', empty_file)