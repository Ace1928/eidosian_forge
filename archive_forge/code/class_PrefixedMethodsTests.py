import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
class PrefixedMethodsTests(TestCase):
    """
    Tests for L{prefixedMethods} which finds methods on a class hierarchy and
    adds them to a dictionary.
    """

    def test_onlyObject(self):
        """
        L{prefixedMethods} returns a list of the methods discovered on an
        object.
        """
        x = Base()
        output = prefixedMethods(x)
        self.assertEqual([x.method], output)

    def test_prefix(self):
        """
        If a prefix is given, L{prefixedMethods} returns only methods named
        with that prefix.
        """
        x = Separate()
        output = prefixedMethods(x, 'good_')
        self.assertEqual([x.good_method], output)