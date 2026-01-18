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
class SynchronousIterateTestsTests(IterateTestsMixin, unittest.SynchronousTestCase):
    """
    Check that L{_iterateTests} returns a list of all test cases in a test suite
    or test case for synchronous tests.

    See L{twisted.trial.test.test_tests.IterateTestsMixin}
    """
    TestCase = unittest.SynchronousTestCase