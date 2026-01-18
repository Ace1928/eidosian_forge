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
class SynchronousMonkeyPatchTests(MonkeyPatchMixin, unittest.SynchronousTestCase):
    """
    Tests for the patch() helper method in the synchronous case.

    See L{twisted.trial.test.test_tests.MonkeyPatchMixin}
    """
    TestCase = unittest.SynchronousTestCase