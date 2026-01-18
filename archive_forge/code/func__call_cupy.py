import types
import numpy as np
import cupy as cp
from cupyx.fallback_mode import notification
def _call_cupy(func, args, kwargs):
    """
    Calls cupy function with *args and **kwargs and
    does necessary data transfers.

    Args:
        func: A cupy function that needs to be called.
        args (tuple): Arguments.
        kwargs (dict): Keyword arguments.

    Returns:
        Result after calling func and performing data transfers.
    """
    _update_cupy_args(args, kwargs)
    cupy_args, cupy_kwargs = _convert_fallback_to_cupy(args, kwargs)
    cupy_res = func(*cupy_args, **cupy_kwargs)
    ext_res = _get_same_reference(cupy_res, cupy_args, cupy_kwargs, args, kwargs)
    if ext_res is not None:
        return ext_res
    if isinstance(cupy_res, cp.ndarray):
        if cupy_res.base is None:
            fallback_res = _convert_cupy_to_fallback(cupy_res)
        else:
            base_arg = _get_same_reference(cupy_res.base, cupy_args, cupy_kwargs, args, kwargs)
            fallback_res = _convert_cupy_to_fallback(cupy_res)
            fallback_res.base = base_arg
        return fallback_res
    return cupy_res