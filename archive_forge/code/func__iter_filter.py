import functools
import inspect
import warnings
from collections import OrderedDict
from typing import Any, List, Optional, Tuple
import torch
import torch._C as _C
import torch._functorch as _functorch
import torch.utils.hooks as hooks
from torch._C import _functions
from torch._functorch.autograd_function import custom_function_call
def _iter_filter(condition, allow_unknown=False, condition_msg=None, conversion=None):

    def _iter(obj):
        if conversion is not None:
            obj = conversion(obj)
        if condition(obj):
            yield obj
        elif obj is None:
            return
        elif isinstance(obj, (list, tuple)):
            for o in obj:
                yield from _iter(o)
        elif isinstance(obj, dict):
            for o in obj.values():
                yield from _iter(o)
        elif allow_unknown:
            yield obj
        else:
            raise ValueError("Auto nesting doesn't know how to process an input object of type " + torch.typename(obj) + ('. Accepted types: ' + condition_msg + ', or lists/tuples of them' if condition_msg else ''))
    return _iter