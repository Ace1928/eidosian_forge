import collections
import contextlib
import math
import operator
from functools import cached_property
from llvmlite import ir
from numba.core import types, typing, cgutils
from numba.core.imputils import (lower_builtin, lower_cast,
from numba.misc import quicksort
from numba.cpython import slicing
from numba.core.errors import NumbaValueError, TypingError
from numba.core.extending import overload, overload_method, intrinsic
@contextlib.contextmanager
def _next_entry(self):
    """
        Yield a random entry from the payload.  Caller must ensure the
        set isn't empty, otherwise the function won't end.
        """
    context = self._context
    builder = self._builder
    intp_t = context.get_value_type(types.intp)
    zero = ir.Constant(intp_t, 0)
    one = ir.Constant(intp_t, 1)
    mask = self.mask
    bb_body = builder.append_basic_block('next_entry_body')
    bb_end = builder.append_basic_block('next_entry_end')
    index = cgutils.alloca_once_value(builder, self.finger)
    builder.branch(bb_body)
    with builder.goto_block(bb_body):
        i = builder.load(index)
        i = builder.and_(mask, builder.add(i, one))
        builder.store(i, index)
        entry = self.get_entry(i)
        is_used = is_hash_used(context, builder, entry.hash)
        builder.cbranch(is_used, bb_end, bb_body)
    builder.position_at_end(bb_end)
    i = builder.load(index)
    self.finger = i
    yield self.get_entry(i)