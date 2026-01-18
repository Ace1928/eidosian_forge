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
class AsynchronousFixtureTests(FixtureMixin, unittest.TestCase):
    """
    Tests for broken fixture helper methods in the asynchronous case

    See: L{twisted.trial.test.test_tests.FixtureMixin}
    """
    TestFailureInSetUp = namedAny('twisted.trial.test.erroneous.AsynchronousTestFailureInSetUp')
    TestFailureInTearDown = namedAny('twisted.trial.test.erroneous.AsynchronousTestFailureInTearDown')
    TestFailureButTearDownRuns = namedAny('twisted.trial.test.erroneous.AsynchronousTestFailureButTearDownRuns')