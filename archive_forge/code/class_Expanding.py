from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union
import pandas.core.window.rolling
from pandas.core.dtypes.common import is_list_like
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from modin.pandas.utils import cast_function_modin2pandas
from modin.utils import _inherit_docstrings
@_inherit_docstrings(pandas.core.window.expanding.Expanding, excluded=[pandas.core.window.expanding.Expanding.__init__])
class Expanding(ClassLogger):

    def __init__(self, dataframe, min_periods=1, axis=0, method='single'):
        self._dataframe = dataframe
        self._query_compiler = dataframe._query_compiler
        self.expanding_args = [min_periods, axis, method]
        self.axis = axis

    def aggregate(self, func, *args, **kwargs):
        from .dataframe import DataFrame
        dataframe = DataFrame(query_compiler=self._query_compiler.expanding_aggregate(self.axis, self.expanding_args, func, *args, **kwargs))
        if isinstance(self._dataframe, DataFrame):
            return dataframe
        elif is_list_like(func):
            dataframe.columns = dataframe.columns.droplevel()
            return dataframe
        else:
            return dataframe.squeeze()

    def sum(self, *args, **kwargs):
        return self._dataframe.__constructor__(query_compiler=self._query_compiler.expanding_sum(self.axis, self.expanding_args, *args, **kwargs))

    def min(self, *args, **kwargs):
        return self._dataframe.__constructor__(query_compiler=self._query_compiler.expanding_min(self.axis, self.expanding_args, *args, **kwargs))

    def max(self, *args, **kwargs):
        return self._dataframe.__constructor__(query_compiler=self._query_compiler.expanding_max(self.axis, self.expanding_args, *args, **kwargs))

    def mean(self, *args, **kwargs):
        return self._dataframe.__constructor__(query_compiler=self._query_compiler.expanding_mean(self.axis, self.expanding_args, *args, **kwargs))

    def median(self, numeric_only=False, engine=None, engine_kwargs=None, **kwargs):
        return self._dataframe.__constructor__(query_compiler=self._query_compiler.expanding_median(self.axis, self.expanding_args, numeric_only=numeric_only, engine=engine, engine_kwargs=engine_kwargs, **kwargs))

    def var(self, *args, **kwargs):
        return self._dataframe.__constructor__(query_compiler=self._query_compiler.expanding_var(self.axis, self.expanding_args, *args, **kwargs))

    def std(self, *args, **kwargs):
        return self._dataframe.__constructor__(query_compiler=self._query_compiler.expanding_std(self.axis, self.expanding_args, *args, **kwargs))

    def count(self, *args, **kwargs):
        return self._dataframe.__constructor__(query_compiler=self._query_compiler.expanding_count(self.axis, self.expanding_args, *args, **kwargs))

    def cov(self, other=None, pairwise=None, ddof=1, numeric_only=False, **kwargs):
        from .dataframe import DataFrame
        from .series import Series
        return self._dataframe.__constructor__(query_compiler=self._query_compiler.expanding_cov(self.axis, self.expanding_args, squeeze_self=isinstance(self._dataframe, Series), squeeze_other=isinstance(other, Series), other=other._query_compiler if isinstance(other, (Series, DataFrame)) else other, pairwise=pairwise, ddof=ddof, numeric_only=numeric_only, **kwargs))

    def corr(self, other=None, pairwise=None, ddof=1, numeric_only=False, **kwargs):
        from .dataframe import DataFrame
        from .series import Series
        return self._dataframe.__constructor__(query_compiler=self._query_compiler.expanding_corr(self.axis, self.expanding_args, squeeze_self=isinstance(self._dataframe, Series), squeeze_other=isinstance(other, Series), other=other._query_compiler if isinstance(other, (Series, DataFrame)) else other, pairwise=pairwise, ddof=ddof, numeric_only=numeric_only, **kwargs))

    def sem(self, ddof=1, numeric_only=False, *args, **kwargs):
        return self._dataframe.__constructor__(query_compiler=self._query_compiler.expanding_sem(self.axis, self.expanding_args, *args, ddof=ddof, numeric_only=numeric_only, **kwargs))

    def skew(self, numeric_only=False, **kwargs):
        return self._dataframe.__constructor__(query_compiler=self._query_compiler.expanding_skew(self.axis, self.expanding_args, numeric_only=numeric_only, **kwargs))

    def kurt(self, **kwargs):
        return self._dataframe.__constructor__(query_compiler=self._query_compiler.expanding_kurt(self.axis, self.expanding_args, **kwargs))

    def quantile(self, q, interpolation='linear', **kwargs):
        return self._dataframe.__constructor__(query_compiler=self._query_compiler.expanding_quantile(self.axis, self.expanding_args, q, interpolation, **kwargs))

    def rank(self, method='average', ascending=True, pct=False, numeric_only=False, **kwargs):
        return self._dataframe.__constructor__(query_compiler=self._query_compiler.expanding_rank(self.axis, self.expanding_args, method, ascending, pct, numeric_only, **kwargs))