import operator
import numpy as np
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.extending import (overload, intrinsic, overload_method,
from numba.core.cgutils import is_nonelike
from numba.cpython import unicode
@intrinsic
def _unicode_to_bytes(typingctx, s):
    assert s == types.unicode_type
    sig = bytes_type(s)

    def codegen(context, builder, signature, args):
        return unicode_to_bytes_cast(context, builder, s, bytes_type, args[0])._getvalue()
    return (sig, codegen)