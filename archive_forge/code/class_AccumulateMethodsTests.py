import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
class AccumulateMethodsTests(TestCase):
    """
    Tests for L{accumulateMethods} which finds methods on a class hierarchy and
    adds them to a dictionary.
    """

    def test_ownClass(self):
        """
        If x is and instance of Base and Base defines a method named method,
        L{accumulateMethods} adds an item to the given dictionary with
        C{"method"} as the key and a bound method object for Base.method value.
        """
        x = Base()
        output = {}
        accumulateMethods(x, output)
        self.assertEqual({'method': x.method}, output)

    def test_baseClass(self):
        """
        If x is an instance of Sub and Sub is a subclass of Base and Base
        defines a method named method, L{accumulateMethods} adds an item to the
        given dictionary with C{"method"} as the key and a bound method object
        for Base.method as the value.
        """
        x = Sub()
        output = {}
        accumulateMethods(x, output)
        self.assertEqual({'method': x.method}, output)

    def test_prefix(self):
        """
        If a prefix is given, L{accumulateMethods} limits its results to
        methods beginning with that prefix.  Keys in the resulting dictionary
        also have the prefix removed from them.
        """
        x = Separate()
        output = {}
        accumulateMethods(x, output, 'good_')
        self.assertEqual({'method': x.good_method}, output)