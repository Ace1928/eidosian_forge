import numpy as _np
import functools
from scipy.linalg import _fblas
from scipy.linalg._fblas import *  # noqa: E402, F403
def _get_funcs(names, arrays, dtype, lib_name, fmodule, cmodule, fmodule_name, cmodule_name, alias, ilp64=False):
    """
    Return available BLAS/LAPACK functions.

    Used also in lapack.py. See get_blas_funcs for docstring.
    """
    funcs = []
    unpack = False
    dtype = _np.dtype(dtype)
    module1 = (cmodule, cmodule_name)
    module2 = (fmodule, fmodule_name)
    if isinstance(names, str):
        names = (names,)
        unpack = True
    prefix, dtype, prefer_fortran = find_best_blas_type(arrays, dtype)
    if prefer_fortran:
        module1, module2 = (module2, module1)
    for name in names:
        func_name = prefix + name
        func_name = alias.get(func_name, func_name)
        func = getattr(module1[0], func_name, None)
        module_name = module1[1]
        if func is None:
            func = getattr(module2[0], func_name, None)
            module_name = module2[1]
        if func is None:
            raise ValueError(f'{lib_name} function {func_name} could not be found')
        func.module_name, func.typecode = (module_name, prefix)
        func.dtype = dtype
        if not ilp64:
            func.int_dtype = _np.dtype(_np.intc)
        else:
            func.int_dtype = _np.dtype(_np.int64)
        func.prefix = prefix
        funcs.append(func)
    if unpack:
        return funcs[0]
    else:
        return funcs