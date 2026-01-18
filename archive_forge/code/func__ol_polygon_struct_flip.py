import warnings
import numpy as np
from numba import typed, njit, errors, typeof
from numba.core import types
from numba.experimental import structref
from numba.extending import overload_method, overload_attribute
from numba.tests.support import (
@overload_method(PolygonStructType, 'flip')
def _ol_polygon_struct_flip(self):

    def impl(self):
        if self.value is not None:
            self.value = -self.value
    return impl