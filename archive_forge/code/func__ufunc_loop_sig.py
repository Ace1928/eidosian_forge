import collections
import ctypes
import re
import numpy as np
from numba.core import errors, types
from numba.core.typing.templates import signature
from numba.np import npdatetime_helpers
from numba.core.errors import TypingError
from numba.core.cgutils import is_nonelike   # noqa: F401
def _ufunc_loop_sig(out_tys, in_tys):
    if len(out_tys) == 1:
        return signature(out_tys[0], *in_tys)
    else:
        return signature(types.Tuple(out_tys), *in_tys)