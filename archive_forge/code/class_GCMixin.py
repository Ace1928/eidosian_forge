import gc
import sys
import unittest as pyunit
import weakref
from io import StringIO
from twisted.internet import defer, reactor
from twisted.python.compat import _PYPY
from twisted.python.reflect import namedAny
from twisted.trial import reporter, runner, unittest, util
from twisted.trial._asyncrunner import (
from twisted.trial.test import erroneous
from twisted.trial.test.test_suppression import SuppressionMixin
class GCMixin:
    """
    I provide a few mock tests that log setUp, tearDown, test execution and
    garbage collection. I'm used to test whether gc.collect gets called.
    """

    class BasicTest(unittest.SynchronousTestCase):
        """
        Mock test to run.
        """

        def setUp(self):
            """
            Mock setUp
            """
            self._log('setUp')

        def test_foo(self):
            """
            Mock test case
            """
            self._log('test')

        def tearDown(self):
            """
            Mock tear tearDown
            """
            self._log('tearDown')

    def _log(self, msg):
        """
        Log function
        """
        self._collectCalled.append(msg)

    def collect(self):
        """Fake gc.collect"""
        self._log('collect')

    def setUp(self):
        """
        Setup our test case
        """
        self._collectCalled = []
        self.BasicTest._log = self._log
        self._oldCollect = gc.collect
        gc.collect = self.collect

    def tearDown(self):
        """
        Tear down the test
        """
        gc.collect = self._oldCollect