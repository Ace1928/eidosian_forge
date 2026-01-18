from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
import numpy as np
from numba import config, cuda, njit, types
@cuda_registry.register_attr
class Interval_attrs(AttributeTemplate):
    key = IntervalType

    def resolve_width(self, mod):
        return types.float64