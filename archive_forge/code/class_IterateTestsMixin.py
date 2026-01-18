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
class IterateTestsMixin:
    """
    L{_iterateTests} returns a list of all test cases in a test suite or test
    case.
    """

    def test_iterateTestCase(self):
        """
        L{_iterateTests} on a single test case returns a list containing that
        test case.
        """
        test = self.TestCase()
        self.assertEqual([test], list(_iterateTests(test)))

    def test_iterateSingletonTestSuite(self):
        """
        L{_iterateTests} on a test suite that contains a single test case
        returns a list containing that test case.
        """
        test = self.TestCase()
        suite = runner.TestSuite([test])
        self.assertEqual([test], list(_iterateTests(suite)))

    def test_iterateNestedTestSuite(self):
        """
        L{_iterateTests} returns tests that are in nested test suites.
        """
        test = self.TestCase()
        suite = runner.TestSuite([runner.TestSuite([test])])
        self.assertEqual([test], list(_iterateTests(suite)))

    def test_iterateIsLeftToRightDepthFirst(self):
        """
        L{_iterateTests} returns tests in left-to-right, depth-first order.
        """
        test = self.TestCase()
        suite = runner.TestSuite([runner.TestSuite([test]), self])
        self.assertEqual([test, self], list(_iterateTests(suite)))