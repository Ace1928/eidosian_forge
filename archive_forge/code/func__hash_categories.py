from __future__ import annotations
from datetime import (
from decimal import Decimal
import re
from typing import (
import warnings
import numpy as np
import pytz
from pandas._libs import (
from pandas._libs.interval import Interval
from pandas._libs.properties import cache_readonly
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import (
from pandas._libs.tslibs.offsets import BDay
from pandas.compat import pa_version_under10p1
from pandas.errors import PerformanceWarning
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import (
from pandas.util import capitalize_first_letter
@cache_readonly
def _hash_categories(self) -> int:
    from pandas.core.util.hashing import combine_hash_arrays, hash_array, hash_tuples
    categories = self.categories
    ordered = self.ordered
    if len(categories) and isinstance(categories[0], tuple):
        cat_list = list(categories)
        cat_array = hash_tuples(cat_list)
    else:
        if categories.dtype == 'O' and len({type(x) for x in categories}) != 1:
            hashed = hash((tuple(categories), ordered))
            return hashed
        if DatetimeTZDtype.is_dtype(categories.dtype):
            categories = categories.view('datetime64[ns]')
        cat_array = hash_array(np.asarray(categories), categorize=False)
    if ordered:
        cat_array = np.vstack([cat_array, np.arange(len(cat_array), dtype=cat_array.dtype)])
    else:
        cat_array = np.array([cat_array])
    combined_hashed = combine_hash_arrays(iter(cat_array), num_items=len(cat_array))
    return np.bitwise_xor.reduce(combined_hashed)