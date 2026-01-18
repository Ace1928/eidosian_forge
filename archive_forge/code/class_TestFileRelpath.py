import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
class TestFileRelpath(TestCase):

    def _with_posix_paths(self):
        self.overrideAttr(urlutils, 'local_path_from_url', urlutils._posix_local_path_from_url)
        self.overrideAttr(urlutils, 'MIN_ABS_FILEURL_LENGTH', len('file:///'))
        self.overrideAttr(osutils, 'normpath', osutils._posix_normpath)
        self.overrideAttr(osutils, 'abspath', osutils.posixpath.abspath)
        self.overrideAttr(osutils, 'normpath', osutils._posix_normpath)
        self.overrideAttr(osutils, 'pathjoin', osutils.posixpath.join)
        self.overrideAttr(osutils, 'split', osutils.posixpath.split)
        self.overrideAttr(osutils, 'MIN_ABS_PATHLENGTH', 1)

    def _with_win32_paths(self):
        self.overrideAttr(urlutils, 'local_path_from_url', urlutils._win32_local_path_from_url)
        self.overrideAttr(urlutils, 'MIN_ABS_FILEURL_LENGTH', urlutils.WIN32_MIN_ABS_FILEURL_LENGTH)
        self.overrideAttr(osutils, 'abspath', osutils._win32_abspath)
        self.overrideAttr(osutils, 'normpath', osutils._win32_normpath)
        self.overrideAttr(osutils, 'pathjoin', osutils._win32_pathjoin)
        self.overrideAttr(osutils, 'split', osutils.ntpath.split)
        self.overrideAttr(osutils, 'MIN_ABS_PATHLENGTH', 3)

    def test_same_url_posix(self):
        self._with_posix_paths()
        self.assertEqual('', urlutils.file_relpath('file:///a', 'file:///a'))
        self.assertEqual('', urlutils.file_relpath('file:///a', 'file:///a/'))
        self.assertEqual('', urlutils.file_relpath('file:///a/', 'file:///a'))

    def test_same_url_win32(self):
        self._with_win32_paths()
        self.assertEqual('', urlutils.file_relpath('file:///A:/', 'file:///A:/'))
        self.assertEqual('', urlutils.file_relpath('file:///A|/', 'file:///A:/'))
        self.assertEqual('', urlutils.file_relpath('file:///A:/b/', 'file:///A:/b/'))
        self.assertEqual('', urlutils.file_relpath('file:///A:/b', 'file:///A:/b/'))
        self.assertEqual('', urlutils.file_relpath('file:///A:/b/', 'file:///A:/b'))

    def test_child_posix(self):
        self._with_posix_paths()
        self.assertEqual('b', urlutils.file_relpath('file:///a', 'file:///a/b'))
        self.assertEqual('b', urlutils.file_relpath('file:///a/', 'file:///a/b'))
        self.assertEqual('b/c', urlutils.file_relpath('file:///a', 'file:///a/b/c'))

    def test_child_win32(self):
        self._with_win32_paths()
        self.assertEqual('b', urlutils.file_relpath('file:///A:/', 'file:///A:/b'))
        self.assertEqual('b', urlutils.file_relpath('file:///A|/', 'file:///A:/b'))
        self.assertEqual('c', urlutils.file_relpath('file:///A:/b', 'file:///A:/b/c'))
        self.assertEqual('c', urlutils.file_relpath('file:///A:/b/', 'file:///A:/b/c'))
        self.assertEqual('c/d', urlutils.file_relpath('file:///A:/b', 'file:///A:/b/c/d'))

    def test_sibling_posix(self):
        self._with_posix_paths()
        self.assertRaises(PathNotChild, urlutils.file_relpath, 'file:///a/b', 'file:///a/c')
        self.assertRaises(PathNotChild, urlutils.file_relpath, 'file:///a/b/', 'file:///a/c')
        self.assertRaises(PathNotChild, urlutils.file_relpath, 'file:///a/b/', 'file:///a/c/')

    def test_sibling_win32(self):
        self._with_win32_paths()
        self.assertRaises(PathNotChild, urlutils.file_relpath, 'file:///A:/b', 'file:///A:/c')
        self.assertRaises(PathNotChild, urlutils.file_relpath, 'file:///A:/b/', 'file:///A:/c')
        self.assertRaises(PathNotChild, urlutils.file_relpath, 'file:///A:/b/', 'file:///A:/c/')

    def test_parent_posix(self):
        self._with_posix_paths()
        self.assertRaises(PathNotChild, urlutils.file_relpath, 'file:///a/b', 'file:///a')
        self.assertRaises(PathNotChild, urlutils.file_relpath, 'file:///a/b', 'file:///a/')

    def test_parent_win32(self):
        self._with_win32_paths()
        self.assertRaises(PathNotChild, urlutils.file_relpath, 'file:///A:/b', 'file:///A:/')
        self.assertRaises(PathNotChild, urlutils.file_relpath, 'file:///A:/b/c', 'file:///A:/b')