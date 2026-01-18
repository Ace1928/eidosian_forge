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
def _type_me(self, argtys, kwtys):
    """
        Implement AbstractTemplate.generic() for the typing class
        built by DUFunc._install_type().

        Return the call-site signature after either validating the
        element-wise signature or compiling for it.
        """
    assert not kwtys
    ufunc = self.ufunc
    _handle_inputs_result = npydecl.Numpy_rules_ufunc._handle_inputs(ufunc, argtys, kwtys)
    base_types, explicit_outputs, ndims, layout = _handle_inputs_result
    explicit_output_count = len(explicit_outputs)
    if explicit_output_count > 0:
        ewise_types = tuple(base_types[:-len(explicit_outputs)])
    else:
        ewise_types = tuple(base_types)
    sig, cres = self.find_ewise_function(ewise_types)
    if sig is None:
        if self._frozen:
            raise TypeError('cannot call %s with types %s' % (self, argtys))
        self._compile_for_argtys(ewise_types)
        sig, cres = self.find_ewise_function(ewise_types)
        assert sig is not None
    if explicit_output_count > 0:
        outtys = list(explicit_outputs)
    elif ufunc.nout == 1:
        if ndims > 0:
            outtys = [types.Array(sig.return_type, ndims, layout)]
        else:
            outtys = [sig.return_type]
    else:
        raise NotImplementedError('typing gufuncs (nout > 1)')
    outtys.extend(argtys)
    return signature(*outtys)