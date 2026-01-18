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
def _fix_stencil_index_offsets(self, options):
    """
        Extract the tuple representing the stencil index offsets
        from the program IR to provide to StencilFunc.
        """
    offset_tuple = get_definition(self.func_ir, options['index_offsets'])
    require(hasattr(offset_tuple, 'items'))
    options['index_offsets'] = tuple(offset_tuple.items)
    return True