import __future__  # noqa: F404
import collections
import functools
import types
import warnings
from typing import Dict, Set, List, Any, Callable, Iterable, Type, Tuple
from functools import wraps
import contextlib
import torch
from torch._C import (
def _disable_user_warnings(func: Callable, regex: str='.*is deprecated, please use.*', module: str='torch') -> Callable:
    """
    Decorator that temporarily disables ``UserWarning``s for the given ``module`` if the warning message matches the
    given ``regex`` pattern.

    Arguments
    ---------
    func : function
        Function to disable the warnings for.
    regex : str
        A regex pattern compilable by ``re.compile``. This is used to match the ``UserWarning`` message.
    module : str
        The python module to which the filtering should be restricted.

    Returns
    -------
    function
        The wrapped function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, message=regex, module=module)
            return func(*args, **kwargs)
    return wrapper