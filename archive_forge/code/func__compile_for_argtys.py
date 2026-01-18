import functools
import warnings
import numpy as np
from numba import jit, typeof
from numba.core import cgutils, types, serialize, sigutils, errors
from numba.core.extending import (is_jitted, overload_attribute,
from numba.core.typing import npydecl
from numba.core.typing.templates import AbstractTemplate, signature
from numba.cpython.unsafe.tuple import tuple_setitem
from numba.np.ufunc import _internal
from numba.parfors import array_analysis
from numba.np.ufunc import ufuncbuilder
from numba.np import numpy_support
from typing import Callable
from llvmlite import ir
def _compile_for_argtys(self, argtys, return_type=None):
    """
        Given a tuple of argument types (these should be the array
        dtypes, and not the array types themselves), compile the
        element-wise function for those inputs, generate a UFunc loop
        wrapper, and register the loop with the Numpy ufunc object for
        this DUFunc.
        """
    if self._frozen:
        raise RuntimeError('compilation disabled for %s' % (self,))
    assert isinstance(argtys, tuple)
    if return_type is None:
        sig = argtys
    else:
        sig = return_type(*argtys)
    cres, argtys, return_type = ufuncbuilder._compile_element_wise_function(self._dispatcher, self.targetoptions, sig)
    actual_sig = ufuncbuilder._finalize_ufunc_signature(cres, argtys, return_type)
    dtypenums, ptr, env = ufuncbuilder._build_element_wise_ufunc_wrapper(cres, actual_sig)
    self._add_loop(int(ptr), dtypenums)
    self._keepalive.append((ptr, cres.library, env))
    self._lower_me.libs.append(cres.library)
    return cres