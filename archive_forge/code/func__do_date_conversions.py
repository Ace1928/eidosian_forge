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
def _do_date_conversions(self, names: Sequence[Hashable] | Index, data: Mapping[Hashable, ArrayLike] | DataFrame) -> tuple[Sequence[Hashable] | Index, Mapping[Hashable, ArrayLike] | DataFrame]:
    if self.parse_dates is not None:
        data, names = _process_date_conversion(data, self._date_conv, self.parse_dates, self.index_col, self.index_names, names, keep_date_col=self.keep_date_col, dtype_backend=self.dtype_backend)
    return (names, data)