import contextlib
import copy
import functools
import inspect
import os
import re
import warnings
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar
from typing_extensions import ParamSpec
import torch
from torch._jit_internal import (
from torch.autograd import function
from torch.jit._script import _CachedForward, script, ScriptModule
from torch.jit._state import _enabled, _python_cu
from torch.nn import Module
from torch.testing._comparison import default_tolerances
def _script_if_tracing(fn: Callable[P, R]) -> Callable[P, R]:

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if not is_tracing():
            return fn(*args, **kwargs)
        compiled_fn: Callable[P, R] = script(wrapper.__original_fn)
        return compiled_fn(*args, **kwargs)
    wrapper.__original_fn = fn
    wrapper.__script_if_tracing_wrapper = True
    return wrapper