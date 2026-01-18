import warnings
import numpy as np
import operator
from numba.core import types, utils, config
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.np.numpy_support import (ufunc_find_matching_loop,
from numba.core.errors import (TypingError, NumbaPerformanceWarning,
from numba import pndindex
@infer_global(operator.eq)
class DtypeEq(AbstractTemplate):

    def generic(self, args, kws):
        [lhs, rhs] = args
        if isinstance(lhs, types.DType) and isinstance(rhs, types.DType):
            return signature(types.boolean, lhs, rhs)