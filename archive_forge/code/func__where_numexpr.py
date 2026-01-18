from __future__ import annotations
import operator
from typing import TYPE_CHECKING
import warnings
import numpy as np
from pandas._config import get_option
from pandas.util._exceptions import find_stack_level
from pandas.core import roperator
from pandas.core.computation.check import NUMEXPR_INSTALLED
def _where_numexpr(cond, a, b):
    result = None
    if _can_use_numexpr(None, 'where', a, b, 'where'):
        result = ne.evaluate('where(cond_value, a_value, b_value)', local_dict={'cond_value': cond, 'a_value': a, 'b_value': b}, casting='safe')
    if result is None:
        result = _where_standard(cond, a, b)
    return result