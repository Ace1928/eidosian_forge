import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
def _checkFullyQualifiedName(self, obj, expected):
    """
        Helper to check that fully qualified name of C{obj} results to
        C{expected}.
        """
    self.assertEqual(fullyQualifiedName(obj), expected)