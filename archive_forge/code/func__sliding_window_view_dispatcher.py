import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.core.overrides import array_function_dispatch, set_module
def _sliding_window_view_dispatcher(x, window_shape, axis=None, *, subok=None, writeable=None):
    return (x,)