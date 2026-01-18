import warnings
import dis
from itertools import product
import numpy as np
from numba import njit, typed, objmode, prange
from numba.core.utils import PYVERSION
from numba.core import ir_utils, ir
from numba.core.errors import (
from numba.tests.support import (
def _multi_inner(self):

    @njit
    def inner(x):
        if x == 1:
            print('call_one')
            raise MyError('one')
        elif x == 2:
            print('call_two')
            raise MyError('two')
        elif x == 3:
            print('call_three')
            raise MyError('three')
        else:
            print('call_other')
    return inner