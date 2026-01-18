from __future__ import annotations
import functools
import operator
import re
import textwrap
from typing import (
import unicodedata
import numpy as np
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas.compat import (
from pandas.util._decorators import doc
from pandas.util._validators import validate_fillna_kwargs
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.missing import isna
from pandas.core import (
from pandas.core.algorithms import map_array
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._arrow_string_mixins import ArrowStringArrayMixin
from pandas.core.arrays._utils import to_numpy_dtype_inference
from pandas.core.arrays.base import (
from pandas.core.arrays.masked import BaseMaskedArray
from pandas.core.arrays.string_ import StringDtype
import pandas.core.common as com
from pandas.core.indexers import (
from pandas.core.strings.base import BaseStringArrayMethods
from pandas.io._util import _arrow_dtype_mapping
from pandas.tseries.frequencies import to_offset
def _str_replace(self, pat: str | re.Pattern, repl: str | Callable, n: int=-1, case: bool=True, flags: int=0, regex: bool=True):
    if isinstance(pat, re.Pattern) or callable(repl) or (not case) or flags:
        raise NotImplementedError('replace is not supported with a re.Pattern, callable repl, case=False, or flags!=0')
    func = pc.replace_substring_regex if regex else pc.replace_substring
    pa_max_replacements = None if n < 0 else n
    result = func(self._pa_array, pattern=pat, replacement=repl, max_replacements=pa_max_replacements)
    return type(self)(result)