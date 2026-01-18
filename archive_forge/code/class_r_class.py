from __future__ import absolute_import
import types
import warnings
from autograd.extend import primitive, notrace_primitive
import numpy as _np
import autograd.builtins as builtins
from numpy.core.einsumfunc import _parse_einsum_input
class r_class:

    def __getitem__(self, args):
        raw_array = _np.r_[args]
        return wrap_if_boxes_inside(raw_array, slow_op_name='r_')