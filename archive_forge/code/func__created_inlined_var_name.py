import types as pytypes  # avoid confusion with numba.types
import copy
import ctypes
import numba.core.analysis
from numba.core import types, typing, errors, ir, rewrites, config, ir_utils
from numba.parfors.parfor import internal_prange
from numba.core.ir_utils import (
from numba.core.analysis import (
from numba.core import postproc
from numba.np.unsafe.ndarray import empty_inferred as unsafe_empty_inferred
import numpy as np
import operator
import numba.misc.special
def _created_inlined_var_name(function_name, var_name):
    """Creates a name for an inlined variable based on the function name and the
    variable name. It does this "safely" to avoid the use of characters that are
    illegal in python variable names as there are occasions when function
    generation needs valid python name tokens."""
    inlined_name = f'{function_name}.{var_name}'
    new_name = inlined_name.replace('<', '_').replace('>', '_')
    new_name = new_name.replace('.', '_').replace('$', '_v')
    return new_name