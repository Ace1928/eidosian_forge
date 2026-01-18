from functools import reduce
import operator
import math
from llvmlite import ir
import llvmlite.binding as ll
from numba.core.imputils import Registry, lower_cast
from numba.core.typing.npydecl import parse_dtype
from numba.core.datamodel import models
from numba.core import types, cgutils
from numba.np import ufunc_db
from numba.np.npyimpl import register_ufuncs
from .cudadrv import nvvm
from numba import cuda
from numba.cuda import nvvmutils, stubs, errors
from numba.cuda.types import dim3, CUDADispatcher
def _atomic_dispatcher(dispatch_fn):

    def imp(context, builder, sig, args):
        aryty, indty, valty = sig.args
        ary, inds, val = args
        dtype = aryty.dtype
        indty, indices = _normalize_indices(context, builder, indty, inds, aryty, valty)
        lary = context.make_array(aryty)(context, builder, ary)
        ptr = cgutils.get_item_pointer(context, builder, aryty, lary, indices, wraparound=True)
        return dispatch_fn(context, builder, dtype, ptr, val)
    return imp