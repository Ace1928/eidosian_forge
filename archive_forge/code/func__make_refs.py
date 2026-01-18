import functools
import math
import operator
from llvmlite import ir
from llvmlite.ir import Constant
import numpy as np
from numba import pndindex, literal_unroll
from numba.core import types, typing, errors, cgutils, extending
from numba.np.numpy_support import (as_dtype, from_dtype, carray, farray,
from numba.np.numpy_support import type_can_asarray, is_nonelike, numpy_version
from numba.core.imputils import (lower_builtin, lower_getattr,
from numba.core.typing import signature
from numba.core.types import StringLiteral
from numba.core.extending import (register_jitable, overload, overload_method,
from numba.misc import quicksort, mergesort
from numba.cpython import slicing
from numba.cpython.unsafe.tuple import tuple_setitem, build_full_slice_tuple
from numba.core.extending import overload_classmethod
from numba.core.typing.npydecl import (parse_dtype as ty_parse_dtype,
def _make_refs(self, ref):
    sig = signature(real_array_type, array_type)
    try:
        array_impl = self._context.get_function('__array__', sig)
    except NotImplementedError:
        return super(ArrayStruct, self)._make_refs(ref)
    datamodel = self._context.data_model_manager[array_type]
    be_type = self._get_be_type(datamodel)
    if ref is None:
        outer_ref = cgutils.alloca_once(self._builder, be_type, zfill=True)
    else:
        outer_ref = ref
    ref = array_impl(self._builder, (outer_ref,))
    return (outer_ref, ref)