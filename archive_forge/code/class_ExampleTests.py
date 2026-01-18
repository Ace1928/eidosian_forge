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
class ExampleTests(tests.TestCase):

    def test_fail(self):
        mutter('this was a failing test')
        self.fail('this test will fail')

    def test_error(self):
        mutter('this test errored')
        raise RuntimeError('gotcha')

    def test_missing_feature(self):
        mutter('missing the feature')
        self.requireFeature(missing_feature)

    def test_skip(self):
        mutter('this test will be skipped')
        raise tests.TestSkipped('reason')

    def test_success(self):
        mutter('this test succeeds')

    def test_xfail(self):
        mutter('test with expected failure')
        self.knownFailure('this_fails')

    def test_unexpected_success(self):
        mutter('test with unexpected success')
        self.expectFailure('should_fail', lambda: None)