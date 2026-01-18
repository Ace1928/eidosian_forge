import types
import numpy as np
import cupy as cp
from cupyx.fallback_mode import notification
def _get_same_reference(res, args, kwargs, ret_args, ret_kwargs):
    """
    Returns object corresponding to res in (args, kwargs)
    from (ret_args, ret_kwargs)
    """
    for i in range(len(args)):
        if res is args[i]:
            return ret_args[i]
    for key in kwargs:
        if res is kwargs[key]:
            return ret_kwargs[key]
    return