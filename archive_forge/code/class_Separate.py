import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
class Separate:
    """
    A no-op class with methods with differing prefixes.
    """

    def good_method(self):
        """
        A no-op method which a matching prefix to be discovered.
        """

    def bad_method(self):
        """
        A no-op method with a mismatched prefix to not be discovered.
        """