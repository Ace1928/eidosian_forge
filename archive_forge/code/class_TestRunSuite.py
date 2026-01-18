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
class TestRunSuite(tests.TestCase):

    def test_runner_class(self):
        """run_suite accepts and uses a runner_class keyword argument."""

        class Stub(tests.TestCase):

            def test_foo(self):
                pass
        suite = Stub('test_foo')
        calls = []

        class MyRunner(tests.TextTestRunner):

            def run(self, test):
                calls.append(test)
                return tests.ExtendedTestResult(self.stream, self.descriptions, self.verbosity)
        tests.run_suite(suite, runner_class=MyRunner, stream=StringIO())
        self.assertLength(1, calls)