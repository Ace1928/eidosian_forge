from __future__ import annotations
from collections.abc import (
from typing import (
import numpy as np
from pandas._libs import lib
from pandas.core.dtypes.common import (
def _check_for_invalid_keys(fname, kwargs, compat_args) -> None:
    """
    Checks whether 'kwargs' contains any keys that are not
    in 'compat_args' and raises a TypeError if there is one.
    """
    diff = set(kwargs) - set(compat_args)
    if diff:
        bad_arg = next(iter(diff))
        raise TypeError(f"{fname}() got an unexpected keyword argument '{bad_arg}'")