from numbers import Number
from statistics import NormalDist
import numpy as np
import pandas as pd
from .algorithms import bootstrap
from .utils import _check_argument
def _validate_errorbar_arg(arg):
    """Check type and value of errorbar argument and assign default level."""
    DEFAULT_LEVELS = {'ci': 95, 'pi': 95, 'se': 1, 'sd': 1}
    usage = '`errorbar` must be a callable, string, or (string, number) tuple'
    if arg is None:
        return (None, None)
    elif callable(arg):
        return (arg, None)
    elif isinstance(arg, str):
        method = arg
        level = DEFAULT_LEVELS.get(method, None)
    else:
        try:
            method, level = arg
        except (ValueError, TypeError) as err:
            raise err.__class__(usage) from err
    _check_argument('errorbar', list(DEFAULT_LEVELS), method)
    if level is not None and (not isinstance(level, Number)):
        raise TypeError(usage)
    return (method, level)