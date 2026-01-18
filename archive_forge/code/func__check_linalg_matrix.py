import warnings
import numpy as np
import operator
from numba.core import types, utils, config
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.np.numpy_support import (ufunc_find_matching_loop,
from numba.core.errors import (TypingError, NumbaPerformanceWarning,
from numba import pndindex
def _check_linalg_matrix(a, func_name):
    if not isinstance(a, types.Array):
        return
    if not a.ndim == 2:
        raise TypingError('np.linalg.%s() only supported on 2-D arrays' % func_name)
    if not isinstance(a.dtype, (types.Float, types.Complex)):
        raise TypingError('np.linalg.%s() only supported on float and complex arrays' % func_name)