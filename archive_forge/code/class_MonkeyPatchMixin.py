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
class MonkeyPatchMixin:
    """
    Tests for the patch() helper method in L{unittest.TestCase}.
    """

    def setUp(self):
        """
        Setup our test case
        """
        self.originalValue = 'original'
        self.patchedValue = 'patched'
        self.objectToPatch = self.originalValue
        self.test = self.TestCase()

    def test_patch(self):
        """
        Calling C{patch()} on a test monkey patches the specified object and
        attribute.
        """
        self.test.patch(self, 'objectToPatch', self.patchedValue)
        self.assertEqual(self.objectToPatch, self.patchedValue)

    def test_patchRestoredAfterRun(self):
        """
        Any monkey patches introduced by a test using C{patch()} are reverted
        after the test has run.
        """
        self.test.patch(self, 'objectToPatch', self.patchedValue)
        self.test.run(reporter.Reporter())
        self.assertEqual(self.objectToPatch, self.originalValue)

    def test_revertDuringTest(self):
        """
        C{patch()} return a L{monkey.MonkeyPatcher} object that can be used to
        restore the original values before the end of the test.
        """
        patch = self.test.patch(self, 'objectToPatch', self.patchedValue)
        patch.restore()
        self.assertEqual(self.objectToPatch, self.originalValue)

    def test_revertAndRepatch(self):
        """
        The returned L{monkey.MonkeyPatcher} object can re-apply the patch
        during the test run.
        """
        patch = self.test.patch(self, 'objectToPatch', self.patchedValue)
        patch.restore()
        patch.patch()
        self.assertEqual(self.objectToPatch, self.patchedValue)

    def test_successivePatches(self):
        """
        Successive patches are applied and reverted just like a single patch.
        """
        self.test.patch(self, 'objectToPatch', self.patchedValue)
        self.assertEqual(self.objectToPatch, self.patchedValue)
        self.test.patch(self, 'objectToPatch', 'second value')
        self.assertEqual(self.objectToPatch, 'second value')
        self.test.run(reporter.Reporter())
        self.assertEqual(self.objectToPatch, self.originalValue)