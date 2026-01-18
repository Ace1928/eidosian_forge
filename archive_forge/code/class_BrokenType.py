import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
class BrokenType(Breakable, type):
    breakName = False

    @property
    def __name__(self):
        if self.breakName:
            raise RuntimeError('no name')
        return 'BrokenType'