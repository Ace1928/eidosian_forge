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
def _define_paths(self, func, *args, **kwargs):
    if isinstance(func, str):
        fast_path = lambda group: getattr(group, func)(*args, **kwargs)
        slow_path = lambda group: group.apply(lambda x: getattr(x, func)(*args, **kwargs), axis=self.axis)
    else:
        fast_path = lambda group: func(group, *args, **kwargs)
        slow_path = lambda group: group.apply(lambda x: func(x, *args, **kwargs), axis=self.axis)
    return (fast_path, slow_path)