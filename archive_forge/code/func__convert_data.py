from __future__ import annotations
from collections import (
from collections.abc import (
import csv
from io import StringIO
import re
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas.errors import (
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.inference import is_dict_like
from pandas.io.common import (
from pandas.io.parsers.base_parser import (
def _convert_data(self, data: Mapping[Hashable, np.ndarray]) -> Mapping[Hashable, ArrayLike]:
    clean_conv = self._clean_mapping(self.converters)
    clean_dtypes = self._clean_mapping(self.dtype)
    clean_na_values = {}
    clean_na_fvalues = {}
    if isinstance(self.na_values, dict):
        for col in self.na_values:
            na_value = self.na_values[col]
            na_fvalue = self.na_fvalues[col]
            if isinstance(col, int) and col not in self.orig_names:
                col = self.orig_names[col]
            clean_na_values[col] = na_value
            clean_na_fvalues[col] = na_fvalue
    else:
        clean_na_values = self.na_values
        clean_na_fvalues = self.na_fvalues
    return self._convert_to_ndarrays(data, clean_na_values, clean_na_fvalues, self.verbose, clean_conv, clean_dtypes)