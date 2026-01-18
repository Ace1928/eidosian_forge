import fnmatch
import functools
import inspect
import os
import warnings
from io import IOBase
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from torch.utils.data._utils.serialization import DILL_AVAILABLE
def _check_unpickable_fn(fn: Callable):
    """
    Check function is pickable or not.

    If it is a lambda or local function, a UserWarning will be raised. If it's not a callable function, a TypeError will be raised.
    """
    if not callable(fn):
        raise TypeError(f'A callable function is expected, but {type(fn)} is provided.')
    if isinstance(fn, partial):
        fn = fn.func
    if _is_local_fn(fn) and (not DILL_AVAILABLE):
        warnings.warn('Local function is not supported by pickle, please use regular python function or functools.partial instead.')
        return
    if hasattr(fn, '__name__') and fn.__name__ == '<lambda>' and (not DILL_AVAILABLE):
        warnings.warn('Lambda function is not supported by pickle, please use regular python function or functools.partial instead.')
        return