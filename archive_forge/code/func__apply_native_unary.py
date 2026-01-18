import torch
from .core import _map_mt_args_kwargs, _wrap_result
def _apply_native_unary(fn, *args, **kwargs):
    if fn in NATIVE_UNARY_FNS:
        return NATIVE_UNARY_MAP[fn](*args, **kwargs)
    if fn in NATIVE_INPLACE_UNARY_FNS:
        return NATIVE_INPLACE_UNARY_MAP[fn](*args, **kwargs)
    return NotImplemented