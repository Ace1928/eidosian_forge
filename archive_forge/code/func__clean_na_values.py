from __future__ import annotations
from collections import (
import csv
import sys
from textwrap import fill
from typing import (
import warnings
import numpy as np
from pandas._config import using_copy_on_write
from pandas._libs import lib
from pandas._libs.parsers import STR_NA_VALUES
from pandas.errors import (
from pandas.util._decorators import Appender
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend
from pandas.core.dtypes.common import (
from pandas import Series
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import RangeIndex
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import (
from pandas.io.parsers.arrow_parser_wrapper import ArrowParserWrapper
from pandas.io.parsers.base_parser import (
from pandas.io.parsers.c_parser_wrapper import CParserWrapper
from pandas.io.parsers.python_parser import (
def _clean_na_values(na_values, keep_default_na: bool=True, floatify: bool=True):
    na_fvalues: set | dict
    if na_values is None:
        if keep_default_na:
            na_values = STR_NA_VALUES
        else:
            na_values = set()
        na_fvalues = set()
    elif isinstance(na_values, dict):
        old_na_values = na_values.copy()
        na_values = {}
        for k, v in old_na_values.items():
            if not is_list_like(v):
                v = [v]
            if keep_default_na:
                v = set(v) | STR_NA_VALUES
            na_values[k] = v
        na_fvalues = {k: _floatify_na_values(v) for k, v in na_values.items()}
    else:
        if not is_list_like(na_values):
            na_values = [na_values]
        na_values = _stringify_na_values(na_values, floatify)
        if keep_default_na:
            na_values = na_values | STR_NA_VALUES
        na_fvalues = _floatify_na_values(na_values)
    return (na_values, na_fvalues)