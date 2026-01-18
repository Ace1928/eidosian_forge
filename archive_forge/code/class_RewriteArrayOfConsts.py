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
@rewrites.register_rewrite('after-inference')
class RewriteArrayOfConsts(rewrites.Rewrite):
    """The RewriteArrayOfConsts class is responsible for finding
    1D array creations from a constant list, and rewriting it into
    direct initialization of array elements without creating the list.
    """

    def __init__(self, state, *args, **kws):
        self.typingctx = state.typingctx
        super(RewriteArrayOfConsts, self).__init__(*args, **kws)

    def match(self, func_ir, block, typemap, calltypes):
        if len(calltypes) == 0:
            return False
        self.crnt_block = block
        self.new_body = guard(_inline_const_arraycall, block, func_ir, self.typingctx, typemap, calltypes)
        return self.new_body is not None

    def apply(self):
        self.crnt_block.body = self.new_body
        return self.crnt_block