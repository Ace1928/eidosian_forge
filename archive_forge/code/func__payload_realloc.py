import math
import operator
from functools import cached_property
from llvmlite import ir
from numba.core import types, typing, errors, cgutils
from numba.core.imputils import (lower_builtin, lower_cast,
from numba.core.extending import overload_method, overload
from numba.misc import quicksort
from numba.cpython import slicing
from numba import literal_unroll
def _payload_realloc(new_allocated):
    payload_type = context.get_data_type(types.ListPayload(self._ty))
    payload_size = context.get_abi_sizeof(payload_type)
    payload_size -= itemsize
    allocsize, ovf = cgutils.muladd_with_overflow(builder, new_allocated, ir.Constant(intp_t, itemsize), ir.Constant(intp_t, payload_size))
    with builder.if_then(ovf, likely=False):
        context.call_conv.return_user_exc(builder, MemoryError, ('cannot resize list',))
    ptr = context.nrt.meminfo_varsize_realloc_unchecked(builder, self._list.meminfo, size=allocsize)
    cgutils.guard_memory_error(context, builder, ptr, 'cannot resize list')
    self._payload.allocated = new_allocated