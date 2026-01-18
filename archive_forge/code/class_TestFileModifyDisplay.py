from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
class TestFileModifyDisplay(TestCase):

    def test_filemodify_file(self):
        c = commands.FileModifyCommand(b'foo/bar', 33188, b':23', None)
        self.assertEqual(b'M 644 :23 foo/bar', bytes(c))

    def test_filemodify_file_executable(self):
        c = commands.FileModifyCommand(b'foo/bar', 33261, b':23', None)
        self.assertEqual(b'M 755 :23 foo/bar', bytes(c))

    def test_filemodify_file_internal(self):
        c = commands.FileModifyCommand(b'foo/bar', 33188, None, b'hello world')
        self.assertEqual(b'M 644 inline foo/bar\ndata 11\nhello world', bytes(c))

    def test_filemodify_symlink(self):
        c = commands.FileModifyCommand(b'foo/bar', 40960, None, b'baz')
        self.assertEqual(b'M 120000 inline foo/bar\ndata 3\nbaz', bytes(c))

    def test_filemodify_treeref(self):
        c = commands.FileModifyCommand(b'tree-info', 57344, b'revision-id-info', None)
        self.assertEqual(b'M 160000 revision-id-info tree-info', bytes(c))