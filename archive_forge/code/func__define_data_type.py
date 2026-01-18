from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
def _define_data_type(self, data_type):
    if data_type.is_opaque:
        data_type.set_body(self._actual_model.get_data_type())