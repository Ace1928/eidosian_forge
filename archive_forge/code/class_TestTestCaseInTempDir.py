import doctest
import gc
import os
import signal
import sys
import threading
import time
import unittest
import warnings
from functools import reduce
from io import BytesIO, StringIO, TextIOWrapper
import testtools.testresult.doubles
from testtools import ExtendedToOriginalDecorator, MultiTestResult
from testtools.content import Content
from testtools.content_type import ContentType
from testtools.matchers import DocTestMatches, Equals
import breezy
from .. import (branchbuilder, controldir, errors, hooks, lockdir, memorytree,
from ..bzr import (bzrdir, groupcompress_repo, remote, workingtree_3,
from ..git import workingtree as git_workingtree
from ..symbol_versioning import (deprecated_function, deprecated_in,
from ..trace import mutter, note
from ..transport import memory
from . import TestUtil, features, test_lsprof, test_server
class TestTestCaseInTempDir(tests.TestCaseInTempDir):

    def test_home_is_not_working(self):
        self.assertNotEqual(self.test_dir, self.test_home_dir)
        cwd = osutils.getcwd()
        self.assertIsSameRealPath(self.test_dir, cwd)
        self.assertIsSameRealPath(self.test_home_dir, os.environ['HOME'])

    def test_assertEqualStat_equal(self):
        from ..bzr.tests.test_dirstate import _FakeStat
        self.build_tree(['foo'])
        real = os.lstat('foo')
        fake = _FakeStat(real.st_size, real.st_mtime, real.st_ctime, real.st_dev, real.st_ino, real.st_mode)
        self.assertEqualStat(real, fake)

    def test_assertEqualStat_notequal(self):
        self.build_tree(['foo', 'longname'])
        self.assertRaises(AssertionError, self.assertEqualStat, os.lstat('foo'), os.lstat('longname'))

    def test_assertPathExists(self):
        self.assertPathExists('.')
        self.build_tree(['foo/', 'foo/bar'])
        self.assertPathExists('foo/bar')
        self.assertPathDoesNotExist('foo/foo')