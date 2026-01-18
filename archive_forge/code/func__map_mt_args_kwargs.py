import warnings
import torch
from torch.overrides import get_default_nowrap_functions
def _map_mt_args_kwargs(args, kwargs, map_fn):

    def _helper(a, map_fn):
        if is_masked_tensor(a):
            return map_fn(a)
        elif torch.is_tensor(a):
            return a
        elif isinstance(a, list):
            a_impl, _ = _map_mt_args_kwargs(a, {}, map_fn)
            return a_impl
        elif isinstance(a, tuple):
            a_impl, _ = _map_mt_args_kwargs(a, {}, map_fn)
            return tuple(a_impl)
        else:
            return a
    if kwargs is None:
        kwargs = {}
    impl_args = []
    for a in args:
        impl_args.append(_helper(a, map_fn))
    impl_kwargs = {}
    for k in kwargs.keys():
        impl_kwargs[k] = _helper(a, map_fn)
    return (impl_args, impl_kwargs)