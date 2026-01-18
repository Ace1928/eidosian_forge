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
class TestMacFuncsDirs(tests.TestCaseInTempDir):
    """Test mac special functions that require directories."""

    def test_getcwd(self):
        self.requireFeature(features.UnicodeFilenameFeature)
        os.mkdir('Bågfors')
        os.chdir('Bågfors')
        self.assertEndsWith(osutils._mac_getcwd(), 'Bågfors')

    def test_getcwd_nonnorm(self):
        self.requireFeature(features.UnicodeFilenameFeature)
        os.mkdir('Bågfors')
        os.chdir('Bågfors')
        self.assertEndsWith(osutils._mac_getcwd(), 'Bågfors')