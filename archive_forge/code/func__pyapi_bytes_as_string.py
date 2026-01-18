import ctypes
import unittest
from numba.core import types
from numba.core.extending import intrinsic
from numba import jit, njit
from numba.tests.support import captured_stdout
@intrinsic
def _pyapi_bytes_as_string(typingctx, csrc, size):
    sig = types.voidptr(csrc, size)

    def codegen(context, builder, sig, args):
        [csrc, size] = args
        api = context.get_python_api(builder)
        b = api.bytes_from_string_and_size(csrc, size)
        return api.bytes_as_string(b)
    return (sig, codegen)