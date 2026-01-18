from __future__ import annotations
from collections import defaultdict
from copy import copy
import csv
import datetime
from enum import Enum
import itertools
from typing import (
import warnings
import numpy as np
from pandas._libs import (
import pandas._libs.ops as libops
from pandas._libs.parsers import STR_NA_VALUES
from pandas._libs.tslibs import parsing
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.astype import astype_array
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import isna
from pandas import (
from pandas.core import algorithms
from pandas.core.arrays import (
from pandas.core.arrays.boolean import BooleanDtype
from pandas.core.indexes.api import (
from pandas.core.series import Series
from pandas.core.tools import datetimes as tools
from pandas.io.common import is_potential_multi_index
@final
def _get_complex_date_index(self, data, col_names):

    def _get_name(icol):
        if isinstance(icol, str):
            return icol
        if col_names is None:
            raise ValueError(f'Must supply column order to use {icol!s} as index')
        for i, c in enumerate(col_names):
            if i == icol:
                return c
    to_remove = []
    index = []
    for idx in self.index_col:
        name = _get_name(idx)
        to_remove.append(name)
        index.append(data[name])
    for c in sorted(to_remove, reverse=True):
        data.pop(c)
        col_names.remove(c)
    return index