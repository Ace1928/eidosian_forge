from __future__ import annotations
from collections.abc import (
from typing import (
import numpy as np
from pandas._libs import lib
from pandas.core.dtypes.common import (
def _check_for_default_values(fname, arg_val_dict, compat_args) -> None:
    """
    Check that the keys in `arg_val_dict` are mapped to their
    default values as specified in `compat_args`.

    Note that this function is to be called only when it has been
    checked that arg_val_dict.keys() is a subset of compat_args
    """
    for key in arg_val_dict:
        try:
            v1 = arg_val_dict[key]
            v2 = compat_args[key]
            if v1 is not None and v2 is None or (v1 is None and v2 is not None):
                match = False
            else:
                match = v1 == v2
            if not is_bool(match):
                raise ValueError("'match' is not a boolean")
        except ValueError:
            match = arg_val_dict[key] is compat_args[key]
        if not match:
            raise ValueError(f"the '{key}' parameter is not supported in the pandas implementation of {fname}()")