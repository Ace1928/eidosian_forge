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
class GarbageCollectionDefaultTests(GCMixin, unittest.SynchronousTestCase):
    """
    By default, tests should not force garbage collection.
    """

    def test_collectNotDefault(self):
        """
        By default, tests should not force garbage collection.
        """
        test = self.BasicTest('test_foo')
        result = reporter.TestResult()
        test.run(result)
        self.assertEqual(self._collectCalled, ['setUp', 'test', 'tearDown'])