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
class GarbageCollectionTests(GCMixin, unittest.SynchronousTestCase):
    """
    Test that, when force GC, it works.
    """

    def test_collectCalled(self):
        """
        test gc.collect is called before and after each test.
        """
        test = GarbageCollectionTests.BasicTest('test_foo')
        test = _ForceGarbageCollectionDecorator(test)
        result = reporter.TestResult()
        test.run(result)
        self.assertEqual(self._collectCalled, ['collect', 'setUp', 'test', 'tearDown', 'collect'])