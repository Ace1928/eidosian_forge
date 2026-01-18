import os
from typing import List
from .. import osutils, tests, win32utils
from ..win32utils import get_app_path, glob_expand
from . import TestCase, TestCaseInTempDir, TestSkipped, features
from .features import backslashdir_feature
class TestWin32UtilsGlobExpand(TestCaseInTempDir):
    _test_needs_features: List[features.Feature] = []

    def test_empty_tree(self):
        self.build_tree([])
        self._run_testset([[['a'], ['a']], [['?'], ['?']], [['*'], ['*']], [['a', 'a'], ['a', 'a']]])

    def build_ascii_tree(self):
        self.build_tree(['a', 'a1', 'a2', 'a11', 'a.1', 'b', 'b1', 'b2', 'b3', 'c/', 'c/c1', 'c/c2', 'd/', 'd/d1', 'd/d2', 'd/e/', 'd/e/e1'])

    def build_unicode_tree(self):
        self.requireFeature(features.UnicodeFilenameFeature)
        self.build_tree(['ሴ', 'ሴሴ', 'ስ/', 'ስ/ስ'])

    def test_tree_ascii(self):
        """Checks the glob expansion and path separation char
        normalization"""
        self.build_ascii_tree()
        self._run_testset([[['a'], ['a']], [['a', 'a'], ['a', 'a']], [['d'], ['d']], [['d/'], ['d/']], [['a*'], ['a', 'a1', 'a2', 'a11', 'a.1']], [['?'], ['a', 'b', 'c', 'd']], [['a?'], ['a1', 'a2']], [['a??'], ['a11', 'a.1']], [['b[1-2]'], ['b1', 'b2']], [['d/*'], ['d/d1', 'd/d2', 'd/e']], [['?/*'], ['c/c1', 'c/c2', 'd/d1', 'd/d2', 'd/e']], [['*/*'], ['c/c1', 'c/c2', 'd/d1', 'd/d2', 'd/e']], [['*/'], ['c/', 'd/']]])

    def test_backslash_globbing(self):
        self.requireFeature(backslashdir_feature)
        self.build_ascii_tree()
        self._run_testset([[['d\\'], ['d/']], [['d\\*'], ['d/d1', 'd/d2', 'd/e']], [['?\\*'], ['c/c1', 'c/c2', 'd/d1', 'd/d2', 'd/e']], [['*\\*'], ['c/c1', 'c/c2', 'd/d1', 'd/d2', 'd/e']], [['*\\'], ['c/', 'd/']]])

    def test_case_insensitive_globbing(self):
        if os.path.normcase('AbC') == 'AbC':
            self.skipTest('Test requires case insensitive globbing function')
        self.build_ascii_tree()
        self._run_testset([[['A'], ['A']], [['A?'], ['a1', 'a2']]])

    def test_tree_unicode(self):
        """Checks behaviour with non-ascii filenames"""
        self.build_unicode_tree()
        self._run_testset([[['ሴ'], ['ሴ']], [['ስ'], ['ስ']], [['ስ/'], ['ስ/']], [['ስ/ስ'], ['ስ/ስ']], [['?'], ['ሴ', 'ስ']], [['*'], ['ሴ', 'ሴሴ', 'ስ']], [['ሴ*'], ['ሴ', 'ሴሴ']], [['ስ/?'], ['ስ/ስ']], [['ስ/*'], ['ስ/ስ']], [['?/'], ['ስ/']], [['*/'], ['ስ/']], [['?/?'], ['ስ/ስ']], [['*/*'], ['ስ/ስ']]])

    def test_unicode_backslashes(self):
        self.requireFeature(backslashdir_feature)
        self.build_unicode_tree()
        self._run_testset([[['ስ\\'], ['ስ/']], [['ስ\\ስ'], ['ስ/ስ']], [['ስ\\?'], ['ስ/ስ']], [['ስ\\*'], ['ስ/ስ']], [['?\\'], ['ስ/']], [['*\\'], ['ስ/']], [['?\\?'], ['ስ/ስ']], [['*\\*'], ['ስ/ስ']]])

    def _run_testset(self, testset):
        for pattern, expected in testset:
            result = glob_expand(pattern)
            expected.sort()
            result.sort()
            self.assertEqual(expected, result, 'pattern %s' % pattern)