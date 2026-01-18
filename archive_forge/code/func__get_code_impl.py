import operator
import numpy as np
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.extending import (overload, intrinsic, overload_method,
from numba.core.cgutils import is_nonelike
from numba.cpython import unicode
def _get_code_impl(a):
    if isinstance(a, types.CharSeq):
        return charseq_get_code
    elif isinstance(a, types.Bytes):
        return bytes_get_code
    elif isinstance(a, types.UnicodeCharSeq):
        return unicode_charseq_get_code
    elif isinstance(a, types.UnicodeType):
        return unicode_get_code