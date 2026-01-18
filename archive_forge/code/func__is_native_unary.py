import torch
from .core import _map_mt_args_kwargs, _wrap_result
def _is_native_unary(fn):
    return fn in NATIVE_UNARY_FNS or fn in NATIVE_INPLACE_UNARY_FNS