import os
import re
import unittest
from distutils import debug
from distutils.log import WARN
from distutils.errors import DistutilsTemplateError
from distutils.filelist import glob_to_re, translate_pattern, FileList
from distutils import filelist
from test.support import os_helper
from test.support import captured_stdout
from distutils.tests import support
class FindAllTestCase(unittest.TestCase):

    @os_helper.skip_unless_symlink
    def test_missing_symlink(self):
        with os_helper.temp_cwd():
            os.symlink('foo', 'bar')
            self.assertEqual(filelist.findall(), [])

    def test_basic_discovery(self):
        """
        When findall is called with no parameters or with
        '.' as the parameter, the dot should be omitted from
        the results.
        """
        with os_helper.temp_cwd():
            os.mkdir('foo')
            file1 = os.path.join('foo', 'file1.txt')
            os_helper.create_empty_file(file1)
            os.mkdir('bar')
            file2 = os.path.join('bar', 'file2.txt')
            os_helper.create_empty_file(file2)
            expected = [file2, file1]
            self.assertEqual(sorted(filelist.findall()), expected)

    def test_non_local_discovery(self):
        """
        When findall is called with another path, the full
        path name should be returned.
        """
        with os_helper.temp_dir() as temp_dir:
            file1 = os.path.join(temp_dir, 'file1.txt')
            os_helper.create_empty_file(file1)
            expected = [file1]
            self.assertEqual(filelist.findall(temp_dir), expected)