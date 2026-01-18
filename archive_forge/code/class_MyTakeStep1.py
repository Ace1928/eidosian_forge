import copy
from numpy.testing import (assert_almost_equal, assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import cos, sin
from scipy.optimize import basinhopping, OptimizeResult
from scipy.optimize._basinhopping import (
class MyTakeStep1(RandomDisplacement):
    """use a copy of displace, but have it set a special parameter to
    make sure it's actually being used."""

    def __init__(self):
        self.been_called = False
        super().__init__()

    def __call__(self, x):
        self.been_called = True
        return super().__call__(x)