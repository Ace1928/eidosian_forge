from __future__ import annotations
from typing import (
import numpy as np
from numpy import ndarray
from pandas._libs.lib import (
from pandas.errors import UnsupportedFunctionCall
from pandas.util._validators import (
class CompatValidator:

    def __init__(self, defaults, fname=None, method: str | None=None, max_fname_arg_count=None) -> None:
        self.fname = fname
        self.method = method
        self.defaults = defaults
        self.max_fname_arg_count = max_fname_arg_count

    def __call__(self, args, kwargs, fname=None, max_fname_arg_count=None, method: str | None=None) -> None:
        if not args and (not kwargs):
            return None
        fname = self.fname if fname is None else fname
        max_fname_arg_count = self.max_fname_arg_count if max_fname_arg_count is None else max_fname_arg_count
        method = self.method if method is None else method
        if method == 'args':
            validate_args(fname, args, max_fname_arg_count, self.defaults)
        elif method == 'kwargs':
            validate_kwargs(fname, kwargs, self.defaults)
        elif method == 'both':
            validate_args_and_kwargs(fname, args, kwargs, max_fname_arg_count, self.defaults)
        else:
            raise ValueError(f"invalid validation method '{method}'")