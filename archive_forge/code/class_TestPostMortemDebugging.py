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
class TestPostMortemDebugging(tests.TestCase):
    """Check post mortem debugging works when tests fail or error"""

    class TracebackRecordingResult(tests.ExtendedTestResult):

        def __init__(self):
            tests.ExtendedTestResult.__init__(self, StringIO(), 0, 1)
            self.postcode = None

        def _post_mortem(self, tb=None):
            """Record the code object at the end of the current traceback"""
            tb = tb or sys.exc_info()[2]
            if tb is not None:
                next = tb.tb_next
                while next is not None:
                    tb = next
                    next = next.tb_next
                self.postcode = tb.tb_frame.f_code

        def report_error(self, test, err):
            pass

        def report_failure(self, test, err):
            pass

    def test_location_unittest_error(self):
        """Needs right post mortem traceback with erroring unittest case"""

        class Test(unittest.TestCase):

            def runTest(self):
                raise RuntimeError
        result = self.TracebackRecordingResult()
        Test().run(result)
        self.assertEqual(result.postcode, Test.runTest.__code__)

    def test_location_unittest_failure(self):
        """Needs right post mortem traceback with failing unittest case"""

        class Test(unittest.TestCase):

            def runTest(self):
                raise self.failureException
        result = self.TracebackRecordingResult()
        Test().run(result)
        self.assertEqual(result.postcode, Test.runTest.__code__)

    def test_location_bt_error(self):
        """Needs right post mortem traceback with erroring breezy.tests case"""

        class Test(tests.TestCase):

            def test_error(self):
                raise RuntimeError
        result = self.TracebackRecordingResult()
        Test('test_error').run(result)
        self.assertEqual(result.postcode, Test.test_error.__code__)

    def test_location_bt_failure(self):
        """Needs right post mortem traceback with failing breezy.tests case"""

        class Test(tests.TestCase):

            def test_failure(self):
                raise self.failureException
        result = self.TracebackRecordingResult()
        Test('test_failure').run(result)
        self.assertEqual(result.postcode, Test.test_failure.__code__)

    def test_env_var_triggers_post_mortem(self):
        """Check pdb.post_mortem is called iff BRZ_TEST_PDB is set"""
        import pdb
        result = tests.ExtendedTestResult(StringIO(), 0, 1)
        post_mortem_calls = []
        self.overrideAttr(pdb, 'post_mortem', post_mortem_calls.append)
        self.overrideEnv('BRZ_TEST_PDB', None)
        result._post_mortem(1)
        self.overrideEnv('BRZ_TEST_PDB', 'on')
        result._post_mortem(2)
        self.assertEqual([2], post_mortem_calls)