import fnmatch
import functools
import inspect
import os
import warnings
from io import IOBase
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from torch.utils.data._utils.serialization import DILL_AVAILABLE
def _is_local_fn(fn):
    if hasattr(fn, '__code__'):
        return fn.__code__.co_flags & inspect.CO_NESTED
    else:
        if hasattr(fn, '__qualname__'):
            return '<locals>' in fn.__qualname__
        fn_type = type(fn)
        if hasattr(fn_type, '__qualname__'):
            return '<locals>' in fn_type.__qualname__
    return False