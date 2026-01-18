import ctypes
import unittest
from numba.core import types
from numba.core.extending import intrinsic
from numba import jit, njit
from numba.tests.support import captured_stdout
@intrinsic
def _pyapi_bytes_as_string_and_size(typingctx, csrc, size):
    retty = types.Tuple.from_types((csrc, size))
    sig = retty(csrc, size)

    def codegen(context, builder, sig, args):
        [csrc, size] = args
        pyapi = context.get_python_api(builder)
        b = pyapi.bytes_from_string_and_size(csrc, size)
        p_cstr = builder.alloca(pyapi.cstring)
        p_size = builder.alloca(pyapi.py_ssize_t)
        pyapi.bytes_as_string_and_size(b, p_cstr, p_size)
        cstr = builder.load(p_cstr)
        size = builder.load(p_size)
        tup = context.make_tuple(builder, sig.return_type, (cstr, size))
        return tup
    return (sig, codegen)