from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
@property
def _actual_model(self):
    return self._dmm.lookup(self.actual_fe_type)