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
class SupportsSymlinksTests(tests.TestCaseInTempDir):

    def setUp(self):
        super().setUp()
        self.overrideAttr(osutils, '_FILESYSTEM_FINDER', osutils.MtabFilesystemFinder([(b'/usr', 'ext4'), (b'/home', 'vfat'), (b'/home/jelmer/smb', 'ntfs'), (b'/home/jelmer', 'ext2')]))

    def test_returns_bool(self):
        self.assertIsInstance(osutils.supports_symlinks(self.test_dir), bool)

    def test_known(self):
        self.assertTrue(osutils.supports_symlinks('/usr'))
        self.assertFalse(osutils.supports_symlinks('/home/bogus'))
        self.assertTrue(osutils.supports_symlinks('/home/jelmer/osx'))
        self.assertFalse(osutils.supports_symlinks('/home/jelmer/smb'))

    def test_unknown(self):
        have_symlinks = sys.platform != 'win32'
        self.assertIs(osutils.supports_symlinks('/var'), have_symlinks)

    def test_error(self):
        have_symlinks = sys.platform != 'win32'

        def raise_error(path):
            raise errors.DependencyNotPresent('FS', 'TEST')
        self.overrideAttr(osutils, 'get_fs_type', raise_error)
        self.assertIs(osutils.supports_symlinks('/var'), have_symlinks)