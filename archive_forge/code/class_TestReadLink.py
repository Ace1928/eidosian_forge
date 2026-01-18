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
class TestReadLink(tests.TestCaseInTempDir):
    """Exposes os.readlink() problems and the osutils solution.

    The only guarantee offered by os.readlink(), starting with 2.6, is that a
    unicode string will be returned if a unicode string is passed.

    But prior python versions failed to properly encode the passed unicode
    string.
    """
    _test_needs_features = [features.UnicodeFilenameFeature]

    def setUp(self):
        super(tests.TestCaseInTempDir, self).setUp()
        self._test_needs_features.append(features.SymlinkFeature(self.test_dir))
        self.link = 'l€ink'
        self.target = 'targe€t'
        os.symlink(self.target, self.link)

    def test_os_readlink_link_encoding(self):
        self.assertEqual(self.target, os.readlink(self.link))

    def test_os_readlink_link_decoding(self):
        self.assertEqual(os.fsencode(self.target), os.readlink(os.fsencode(self.link)))