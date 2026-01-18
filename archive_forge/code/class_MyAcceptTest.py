import copy
from numpy.testing import (assert_almost_equal, assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import cos, sin
from scipy.optimize import basinhopping, OptimizeResult
from scipy.optimize._basinhopping import (
class MyAcceptTest:
    """pass a custom accept test

    This does nothing but make sure it's being used and ensure all the
    possible return values are accepted
    """

    def __init__(self):
        self.been_called = False
        self.ncalls = 0
        self.testres = [False, 'force accept', True, np.bool_(True), np.bool_(False), [], {}, 0, 1]

    def __call__(self, **kwargs):
        self.been_called = True
        self.ncalls += 1
        if self.ncalls - 1 < len(self.testres):
            return self.testres[self.ncalls - 1]
        else:
            return True