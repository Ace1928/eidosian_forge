import functools
import inspect
import sys
import warnings
import numpy as np
from ._warnings import all_warnings, warn
def _warning_stacklevel(func):
    """Find stacklevel for a warning raised from a wrapper around `func`.

    Try to determine the number of

    Parameters
    ----------
    func : Callable


    Returns
    -------
    stacklevel : int
        The stacklevel. Minimum of 2.
    """
    wrapped_count = _count_wrappers(func)
    module = sys.modules.get(func.__module__)
    try:
        for name in func.__qualname__.split('.'):
            global_func = getattr(module, name)
    except AttributeError as e:
        raise RuntimeError(f'Could not access `{func.__qualname__}` in {module!r},  may be a closure. Set stacklevel manually. ') from e
    else:
        global_wrapped_count = _count_wrappers(global_func)
    stacklevel = global_wrapped_count - wrapped_count + 1
    return max(stacklevel, 2)