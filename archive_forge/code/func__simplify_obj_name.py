import inspect
import functools
from enum import Enum
import torch.autograd
def _simplify_obj_name(obj) -> str:
    """Simplify the display strings of objects for the purpose of rendering within DataPipe error messages."""
    if inspect.isfunction(obj):
        return obj.__name__
    else:
        return repr(obj)