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
def _replace_payload(self, nentries):
    """
        Replace the payload with a new empty payload with the given number
        of entries.

        CAUTION: *nentries* must be a power of 2!
        """
    context = self._context
    builder = self._builder
    with self.payload._iterate() as loop:
        entry = loop.entry
        self.decref_value(entry.key)
    self._free_payload(self.payload.ptr)
    ok = self._allocate_payload(nentries, realloc=True)
    with builder.if_then(builder.not_(ok), likely=False):
        context.call_conv.return_user_exc(builder, MemoryError, ('cannot reallocate set',))