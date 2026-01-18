import warnings
import numpy as np
import operator
from numba.core import types, utils, config
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.np.numpy_support import (ufunc_find_matching_loop,
from numba.core.errors import (TypingError, NumbaPerformanceWarning,
from numba import pndindex
def _numpy_redirect(fname):
    numpy_function = getattr(np, fname)
    cls = type('Numpy_redirect_{0}'.format(fname), (Numpy_method_redirection,), dict(key=numpy_function, method_name=fname))
    infer_global(numpy_function, types.Function(cls))