import torch
from copy import deepcopy
from torch.utils._pytree import tree_map
from torch.testing._internal.logging_tensor import LoggingTensor
@classmethod
def _try_call_special_impl(cls, func, args, kwargs):
    if func not in cls._SPECIAL_IMPLS:
        return NotImplemented
    return cls._SPECIAL_IMPLS[func](args, kwargs)