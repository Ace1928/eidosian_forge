import os
from typing import List
from .. import osutils, tests, win32utils
from ..win32utils import get_app_path, glob_expand
from . import TestCase, TestCaseInTempDir, TestSkipped, features
from .features import backslashdir_feature
class TestLocations(TestCase):
    _test_needs_features = [features.win32_feature]

    def assertPathsEqual(self, p1, p2):
        self.assertEqual(p1, p2)

    def test_appdata_not_using_environment(self):
        first = win32utils.get_appdata_location()
        self.overrideEnv('APPDATA', None)
        self.assertPathsEqual(first, win32utils.get_appdata_location())

    def test_appdata_matches_environment(self):
        encoding = osutils.get_user_encoding()
        env_val = os.environ.get('APPDATA', None)
        if not env_val:
            raise TestSkipped('No APPDATA environment variable exists')
        self.assertPathsEqual(win32utils.get_appdata_location(), env_val.decode(encoding))

    def test_local_appdata_not_using_environment(self):
        first = win32utils.get_local_appdata_location()
        self.overrideEnv('LOCALAPPDATA', None)
        self.assertPathsEqual(first, win32utils.get_local_appdata_location())

    def test_local_appdata_matches_environment(self):
        lad = win32utils.get_local_appdata_location()
        env = os.environ.get('LOCALAPPDATA')
        if env:
            encoding = osutils.get_user_encoding()
            self.assertPathsEqual(lad, env.decode(encoding))