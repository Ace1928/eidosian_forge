import warnings
from typing import Any
import pandas
from pandas.core.dtypes.common import is_list_like
from pandas.core.groupby.base import transformation_kernels
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, hashable
from .default import DefaultMethod
class GroupByDefault(DefaultMethod):
    """Builder for default-to-pandas GroupBy aggregation functions."""
    _groupby_cls = GroupBy
    OBJECT_TYPE = 'GroupBy'

    @classmethod
    def register(cls, func, **kwargs):
        """
        Build default-to-pandas GroupBy aggregation function.

        Parameters
        ----------
        func : callable or str
            Default aggregation function. If aggregation function is not specified
            via groupby arguments, then `func` function is used.
        **kwargs : kwargs
            Additional arguments that will be passed to function builder.

        Returns
        -------
        callable
            Functiom that takes query compiler and defaults to pandas to do GroupBy
            aggregation.
        """
        return super().register(cls._groupby_cls.build_groupby(func), fn_name=func.__name__, **kwargs)
    _aggregation_methods_dict = {'axis_wise': pandas.core.groupby.DataFrameGroupBy.aggregate, 'group_wise': pandas.core.groupby.DataFrameGroupBy.apply, 'transform': pandas.core.groupby.DataFrameGroupBy.transform, 'direct': lambda grp, func, *args, **kwargs: func(grp, *args, **kwargs)}

    @classmethod
    def get_aggregation_method(cls, how):
        """
        Return `pandas.DataFrameGroupBy` method that implements the passed `how` UDF applying strategy.

        Parameters
        ----------
        how : {"axis_wise", "group_wise", "transform"}
            `how` parameter of the ``BaseQueryCompiler.groupby_agg``.

        Returns
        -------
        callable(pandas.DataFrameGroupBy, callable, *args, **kwargs) -> [pandas.DataFrame | pandas.Series]

        Notes
        -----
        Visit ``BaseQueryCompiler.groupby_agg`` doc-string for more information about `how` parameter.
        """
        return cls._aggregation_methods_dict[how]