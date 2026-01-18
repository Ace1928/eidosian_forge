import warnings
import numpy as np
import operator
from numba.core import types, utils, config
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.np.numpy_support import (ufunc_find_matching_loop,
from numba.core.errors import (TypingError, NumbaPerformanceWarning,
from numba import pndindex
def _homogeneous_dims(context, func_name, arrays):
    ndim = arrays[0].ndim
    for a in arrays:
        if a.ndim != ndim:
            msg = f'{func_name}(): all the input arrays must have same number of dimensions'
            raise NumbaTypeError(msg)
    return ndim