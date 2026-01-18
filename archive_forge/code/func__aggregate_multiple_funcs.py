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
def _aggregate_multiple_funcs(self, arg, *args, **kwargs) -> DataFrame:
    if isinstance(arg, dict):
        if self.as_index:
            raise SpecificationError('nested renamer is not supported')
        else:
            msg = 'Passing a dictionary to SeriesGroupBy.agg is deprecated and will raise in a future version of pandas. Pass a list of aggregations instead.'
            warnings.warn(message=msg, category=FutureWarning, stacklevel=find_stack_level())
            arg = list(arg.items())
    elif any((isinstance(x, (tuple, list)) for x in arg)):
        arg = [(x, x) if not isinstance(x, (tuple, list)) else x for x in arg]
    else:
        columns = (com.get_callable_name(f) or f for f in arg)
        arg = zip(columns, arg)
    results: dict[base.OutputKey, DataFrame | Series] = {}
    with com.temp_setattr(self, 'as_index', True):
        for idx, (name, func) in enumerate(arg):
            key = base.OutputKey(label=name, position=idx)
            results[key] = self.aggregate(func, *args, **kwargs)
    if any((isinstance(x, DataFrame) for x in results.values())):
        from pandas import concat
        res_df = concat(results.values(), axis=1, keys=[key.label for key in results])
        return res_df
    indexed_output = {key.position: val for key, val in results.items()}
    output = self.obj._constructor_expanddim(indexed_output, index=None)
    output.columns = Index((key.label for key in results))
    return output