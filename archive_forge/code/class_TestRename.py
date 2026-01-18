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
class TestRename(tests.TestCaseInTempDir):

    def create_file(self, filename, content):
        f = open(filename, 'wb')
        try:
            f.write(content)
        finally:
            f.close()

    def _fancy_rename(self, a, b):
        osutils.fancy_rename(a, b, rename_func=os.rename, unlink_func=os.unlink)

    def test_fancy_rename(self):
        self.create_file('a', b'something in a\n')
        self._fancy_rename('a', 'b')
        self.assertPathDoesNotExist('a')
        self.assertPathExists('b')
        self.check_file_contents('b', b'something in a\n')
        self.create_file('a', b'new something in a\n')
        self._fancy_rename('b', 'a')
        self.check_file_contents('a', b'something in a\n')

    def test_fancy_rename_fails_source_missing(self):
        self.create_file('target', b'data in target\n')
        self.assertRaises((IOError, OSError), self._fancy_rename, 'missingsource', 'target')
        self.assertPathExists('target')
        self.check_file_contents('target', b'data in target\n')

    def test_fancy_rename_fails_if_source_and_target_missing(self):
        self.assertRaises((IOError, OSError), self._fancy_rename, 'missingsource', 'missingtarget')

    def test_rename(self):
        self.create_file('a', b'something in a\n')
        osutils.rename('a', 'b')
        self.assertPathDoesNotExist('a')
        self.assertPathExists('b')
        self.check_file_contents('b', b'something in a\n')
        self.create_file('a', b'new something in a\n')
        osutils.rename('b', 'a')
        self.check_file_contents('a', b'something in a\n')

    def test_rename_change_case(self):
        self.build_tree(['a', 'b/'])
        osutils.rename('a', 'A')
        osutils.rename('b', 'B')
        shape = sorted(os.listdir('.'))
        self.assertEqual(['A', 'B'], shape)

    def test_rename_exception(self):
        try:
            osutils.rename('nonexistent_path', 'different_nonexistent_path')
        except OSError as e:
            self.assertEqual(e.old_filename, 'nonexistent_path')
            self.assertEqual(e.new_filename, 'different_nonexistent_path')
            self.assertTrue('nonexistent_path' in e.strerror)
            self.assertTrue('different_nonexistent_path' in e.strerror)