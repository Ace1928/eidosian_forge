from __future__ import annotations
from collections import abc
from functools import partial
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.hashtable import duplicated
from pandas.errors import SpecificationError
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.inference import is_hashable
from pandas.core.dtypes.missing import (
from pandas.core import algorithms
from pandas.core.apply import (
import pandas.core.common as com
from pandas.core.frame import DataFrame
from pandas.core.groupby import (
from pandas.core.groupby.groupby import (
from pandas.core.indexes.api import (
from pandas.core.series import Series
from pandas.core.sorting import get_group_index
from pandas.core.util.numba_ import maybe_use_numba
from pandas.plotting import boxplot_frame_groupby
def _aggregate_named(self, func, *args, **kwargs):
    result = {}
    initialized = False
    for name, group in self._grouper.get_iterator(self._obj_with_exclusions, axis=self.axis):
        object.__setattr__(group, 'name', name)
        output = func(group, *args, **kwargs)
        output = ops.extract_result(output)
        if not initialized:
            ops.check_result_array(output, group.dtype)
            initialized = True
        result[name] = output
    return result