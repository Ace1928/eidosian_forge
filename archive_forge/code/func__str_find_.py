from __future__ import annotations
import functools
import re
import textwrap
from typing import (
import unicodedata
import numpy as np
from pandas._libs import lib
import pandas._libs.missing as libmissing
import pandas._libs.ops as libops
from pandas.core.dtypes.missing import isna
from pandas.core.strings.base import BaseStringArrayMethods
def _str_find_(self, sub, start, end, side):
    if side == 'left':
        method = 'find'
    elif side == 'right':
        method = 'rfind'
    else:
        raise ValueError('Invalid side')
    if end is None:
        f = lambda x: getattr(x, method)(sub, start)
    else:
        f = lambda x: getattr(x, method)(sub, start, end)
    return self._str_map(f, dtype='int64')