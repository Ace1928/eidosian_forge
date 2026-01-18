import os
from typing import List
from .. import osutils, tests, win32utils
from ..win32utils import get_app_path, glob_expand
from . import TestCase, TestCaseInTempDir, TestSkipped, features
from .features import backslashdir_feature
class TestAppPaths(TestCase):
    _test_needs_features = [Win32RegistryFeature]

    def test_iexplore(self):
        for a in ('iexplore', 'iexplore.exe'):
            p = get_app_path(a)
            d, b = os.path.split(p)
            self.assertEqual('iexplore.exe', b.lower())
            self.assertNotEqual('', d)

    def test_wordpad(self):
        self.requireFeature(Win32ApiFeature)
        for a in ('wordpad', 'wordpad.exe'):
            p = get_app_path(a)
            d, b = os.path.split(p)
            self.assertEqual('wordpad.exe', b.lower())
            self.assertNotEqual('', d)

    def test_not_existing(self):
        p = get_app_path('not-existing')
        self.assertEqual('not-existing', p)