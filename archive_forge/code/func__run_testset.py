import os
from typing import List
from .. import osutils, tests, win32utils
from ..win32utils import get_app_path, glob_expand
from . import TestCase, TestCaseInTempDir, TestSkipped, features
from .features import backslashdir_feature
def _run_testset(self, testset):
    for pattern, expected in testset:
        result = glob_expand(pattern)
        expected.sort()
        result.sort()
        self.assertEqual(expected, result, 'pattern %s' % pattern)