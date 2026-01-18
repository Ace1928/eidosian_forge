import pickle
import pyomo.common.unittest as unittest
from pyomo.core.base.range import (
from pyomo.core.base.set import Any
class _Custom(object):

    def __init__(self, val):
        self.val = val

    def __lt__(self, other):
        return self.val < other

    def __gt__(self, other):
        return self.val > other

    def __le__(self, other):
        return self.val <= other

    def __ge__(self, other):
        return self.val >= other

    def __eq__(self, other):
        return self.val == other

    def __sub__(self, other):
        return self.val - other