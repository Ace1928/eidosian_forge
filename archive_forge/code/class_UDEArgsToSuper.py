import numpy as np
import sys
import traceback
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import (TestCase, expected_failure_py311,
import unittest
class UDEArgsToSuper(Exception):

    def __init__(self, arg, value0):
        super(UDEArgsToSuper, self).__init__(arg)
        self.value0 = value0

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        same = True
        same |= self.args == other.args
        same |= self.value0 == other.value0
        return same

    def __hash__(self):
        return hash((super(UDEArgsToSuper).__hash__(), self.value0))