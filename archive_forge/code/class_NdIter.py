import warnings
import numpy as np
import operator
from numba.core import types, utils, config
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.np.numpy_support import (ufunc_find_matching_loop,
from numba.core.errors import (TypingError, NumbaPerformanceWarning,
from numba import pndindex
@infer_global(np.nditer)
class NdIter(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws
        if len(args) != 1:
            return
        arrays, = args
        if isinstance(arrays, types.BaseTuple):
            if not arrays:
                return
            arrays = list(arrays)
        else:
            arrays = [arrays]
        nditerty = types.NumpyNdIterType(arrays)
        return signature(nditerty, *args)