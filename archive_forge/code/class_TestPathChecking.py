from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
class TestPathChecking(TestCase):

    def test_filemodify_path_checking(self):
        self.assertRaises(ValueError, commands.FileModifyCommand, b'', 33188, None, b'text')
        self.assertRaises(ValueError, commands.FileModifyCommand, None, 33188, None, b'text')

    def test_filedelete_path_checking(self):
        self.assertRaises(ValueError, commands.FileDeleteCommand, b'')
        self.assertRaises(ValueError, commands.FileDeleteCommand, None)

    def test_filerename_path_checking(self):
        self.assertRaises(ValueError, commands.FileRenameCommand, b'', b'foo')
        self.assertRaises(ValueError, commands.FileRenameCommand, None, b'foo')
        self.assertRaises(ValueError, commands.FileRenameCommand, b'foo', b'')
        self.assertRaises(ValueError, commands.FileRenameCommand, b'foo', None)

    def test_filecopy_path_checking(self):
        self.assertRaises(ValueError, commands.FileCopyCommand, b'', b'foo')
        self.assertRaises(ValueError, commands.FileCopyCommand, None, b'foo')
        self.assertRaises(ValueError, commands.FileCopyCommand, b'foo', b'')
        self.assertRaises(ValueError, commands.FileCopyCommand, b'foo', None)