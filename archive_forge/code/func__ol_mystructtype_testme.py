import warnings
import numpy as np
from numba import typed, njit, errors, typeof
from numba.core import types
from numba.experimental import structref
from numba.extending import overload_method, overload_attribute
from numba.tests.support import (
@overload_method(MyStructType, 'testme')
def _ol_mystructtype_testme(self, arg):

    def impl(self, arg):
        return self.values * arg + self.counter
    return impl