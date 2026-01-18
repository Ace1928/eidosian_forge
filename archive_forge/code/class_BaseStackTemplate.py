import warnings
import numpy as np
import operator
from numba.core import types, utils, config
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.np.numpy_support import (ufunc_find_matching_loop,
from numba.core.errors import (TypingError, NumbaPerformanceWarning,
from numba import pndindex
class BaseStackTemplate(CallableTemplate):

    def generic(self):

        def typer(arrays):
            dtype, ndim = _sequence_of_arrays(self.context, self.func_name, arrays)
            ndim = max(ndim, self.ndim_min)
            layout = _choose_concatenation_layout(arrays)
            return types.Array(dtype, ndim, layout)
        return typer