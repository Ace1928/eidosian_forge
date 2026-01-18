import errno
import os
import select
import socket
import sys
import tempfile
import time
from io import BytesIO
from .. import errors, osutils, tests, trace, win32utils
from . import features, file_utils, test__walkdirs_win32
from .scenarios import load_tests_apply_scenarios
class TestWin32FuncsDirs(tests.TestCaseInTempDir):
    """Test win32 functions that create files."""

    def test_getcwd(self):
        self.requireFeature(features.UnicodeFilenameFeature)
        os.mkdir('mu-µ')
        os.chdir('mu-µ')
        self.assertEndsWith(osutils._win32_getcwd(), 'mu-µ')

    def test_minimum_path_selection(self):
        self.assertEqual(set(), osutils.minimum_path_selection([]))
        self.assertEqual({'a'}, osutils.minimum_path_selection(['a']))
        self.assertEqual({'a', 'b'}, osutils.minimum_path_selection(['a', 'b']))
        self.assertEqual({'a/', 'b'}, osutils.minimum_path_selection(['a/', 'b']))
        self.assertEqual({'a/', 'b'}, osutils.minimum_path_selection(['a/c', 'a/', 'b']))
        self.assertEqual({'a-b', 'a', 'a0b'}, osutils.minimum_path_selection(['a-b', 'a/b', 'a0b', 'a']))

    def test_rename(self):
        with open('a', 'wb') as a:
            a.write(b'foo\n')
        with open('b', 'wb') as b:
            b.write(b'baz\n')
        osutils._win32_rename('b', 'a')
        self.assertPathExists('a')
        self.assertPathDoesNotExist('b')
        self.assertFileEqual(b'baz\n', 'a')

    def test_rename_missing_file(self):
        with open('a', 'wb') as a:
            a.write(b'foo\n')
        try:
            osutils._win32_rename('b', 'a')
        except OSError as e:
            self.assertEqual(errno.ENOENT, e.errno)
        self.assertFileEqual(b'foo\n', 'a')

    def test_rename_missing_dir(self):
        os.mkdir('a')
        try:
            osutils._win32_rename('b', 'a')
        except OSError as e:
            self.assertEqual(errno.ENOENT, e.errno)

    def test_rename_current_dir(self):
        os.mkdir('a')
        os.chdir('a')
        try:
            osutils._win32_rename('b', '.')
        except OSError as e:
            self.assertEqual(errno.ENOENT, e.errno)

    def test_splitpath(self):

        def check(expected, path):
            self.assertEqual(expected, osutils.splitpath(path))
        check(['a'], 'a')
        check(['a', 'b'], 'a/b')
        check(['a', 'b'], 'a/./b')
        check(['a', '.b'], 'a/.b')
        if os.path.sep == '\\':
            check(['a', '.b'], 'a\\.b')
        else:
            check(['a\\.b'], 'a\\.b')
        self.assertRaises(errors.BzrError, osutils.splitpath, 'a/../b')