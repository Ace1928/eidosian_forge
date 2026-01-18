from __future__ import annotations
from collections import abc
from typing import (
import warnings
import numpy as np
from pandas._config import using_copy_on_write
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import isna
from pandas.core.arrays.categorical import (
import pandas.core.common as com
from pandas.core.indexes.api import (
from pandas.core.internals import concatenate_managers
def _get_comb_axis(self, i: AxisInt) -> Index:
    data_axis = self.objs[0]._get_block_manager_axis(i)
    return get_objs_combined_axis(self.objs, axis=data_axis, intersect=self.intersect, sort=self.sort, copy=self.copy)