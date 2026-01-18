from __future__ import annotations
import operator
from typing import TYPE_CHECKING
import warnings
import numpy as np
from pandas._config import get_option
from pandas.util._exceptions import find_stack_level
from pandas.core import roperator
from pandas.core.computation.check import NUMEXPR_INSTALLED
def _bool_arith_fallback(op_str, a, b) -> bool:
    """
    Check if we should fallback to the python `_evaluate_standard` in case
    of an unsupported operation by numexpr, which is the case for some
    boolean ops.
    """
    if _has_bool_dtype(a) and _has_bool_dtype(b):
        if op_str in _BOOL_OP_UNSUPPORTED:
            warnings.warn(f'evaluating in Python space because the {repr(op_str)} operator is not supported by numexpr for the bool dtype, use {repr(_BOOL_OP_UNSUPPORTED[op_str])} instead.', stacklevel=find_stack_level())
            return True
    return False