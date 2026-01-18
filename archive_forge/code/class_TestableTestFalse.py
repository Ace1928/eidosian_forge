import datetime
import warnings
import weakref
import unittest
from test.support import gc_collect
from itertools import product
class TestableTestFalse(unittest.TestCase):
    longMessage = False
    failureException = self.failureException

    def testTest(self):
        pass