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
def _normalize_indices(context, builder, indty, inds, aryty, valty):
    """
    Convert integer indices into tuple of intp
    """
    if indty in types.integer_domain:
        indty = types.UniTuple(dtype=indty, count=1)
        indices = [inds]
    else:
        indices = cgutils.unpack_tuple(builder, inds, count=len(indty))
    indices = [context.cast(builder, i, t, types.intp) for t, i in zip(indty, indices)]
    dtype = aryty.dtype
    if dtype != valty:
        raise TypeError('expect %s but got %s' % (dtype, valty))
    if aryty.ndim != len(indty):
        raise TypeError('indexing %d-D array with %d-D index' % (aryty.ndim, len(indty)))
    return (indty, indices)