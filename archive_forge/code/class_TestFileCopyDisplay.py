from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
class TestFileCopyDisplay(TestCase):

    def test_filecopy(self):
        c = commands.FileCopyCommand(b'foo/bar', b'foo/baz')
        self.assertEqual(b'C foo/bar foo/baz', bytes(c))

    def test_filecopy_quoted(self):
        c = commands.FileCopyCommand(b'foo/b a r', b'foo/b a z')
        self.assertEqual(b'C "foo/b a r" foo/b a z', bytes(c))