import warnings
from typing import Any
import pandas
from pandas.core.dtypes.common import is_list_like
from pandas.core.groupby.base import transformation_kernels
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, hashable
from .default import DefaultMethod
class SeriesGroupByDefault(GroupByDefault):
    """Builder for default-to-pandas GroupBy aggregation functions for Series."""
    _groupby_cls = SeriesGroupBy
    _aggregation_methods_dict = {'axis_wise': pandas.core.groupby.SeriesGroupBy.aggregate, 'group_wise': pandas.core.groupby.SeriesGroupBy.apply, 'transform': pandas.core.groupby.SeriesGroupBy.transform, 'direct': lambda grp, func, *args, **kwargs: func(grp, *args, **kwargs)}