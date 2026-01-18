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
class TestFindExecutableInPath(tests.TestCase):

    def test_windows(self):
        if sys.platform != 'win32':
            raise tests.TestSkipped('test requires win32')
        self.assertTrue(osutils.find_executable_on_path('explorer') is not None)
        self.assertTrue(osutils.find_executable_on_path('explorer.exe') is not None)
        self.assertTrue(osutils.find_executable_on_path('EXPLORER.EXE') is not None)
        self.assertTrue(osutils.find_executable_on_path('THIS SHOULD NOT EXIST') is None)
        self.assertTrue(osutils.find_executable_on_path('file.txt') is None)

    def test_windows_app_path(self):
        if sys.platform != 'win32':
            raise tests.TestSkipped('test requires win32')
        self.overrideEnv('PATH', '')
        self.assertTrue(osutils.find_executable_on_path('iexplore') is not None)

    def test_other(self):
        if sys.platform == 'win32':
            raise tests.TestSkipped('test requires non-win32')
        self.assertTrue(osutils.find_executable_on_path('sh') is not None)
        self.assertTrue(osutils.find_executable_on_path('THIS SHOULD NOT EXIST') is None)