from __future__ import annotations
import collections
import itertools as it
import operator
import uuid
import warnings
from functools import partial, wraps
from numbers import Integral
import numpy as np
import pandas as pd
from dask.base import is_dask_collection, tokenize
from dask.core import flatten
from dask.dataframe._compat import (
from dask.dataframe.core import (
from dask.dataframe.dispatch import grouper_dispatch
from dask.dataframe.methods import concat, drop_columns
from dask.dataframe.utils import (
from dask.highlevelgraph import HighLevelGraph
from dask.typing import no_default
from dask.utils import (
class SeriesGroupBy(_GroupBy):
    _token_prefix = 'series-groupby-'

    def __init__(self, df, by=None, slice=None, observed=None, **kwargs):
        observed = {'observed': observed} if observed is not None else {}
        if isinstance(df, Series):
            if isinstance(by, Series):
                pass
            elif isinstance(by, list):
                if len(by) == 0:
                    raise ValueError('No group keys passed!')
                non_series_items = [item for item in by if not isinstance(item, Series)]
                df._meta.groupby(non_series_items, **observed)
            else:
                df._meta.groupby(by, **observed)
        super().__init__(df, by=by, slice=slice, **observed, **kwargs)

    @derived_from(pd.core.groupby.SeriesGroupBy)
    def nunique(self, split_every=None, split_out=1):
        """
        Examples
        --------
        >>> import pandas as pd
        >>> import dask.dataframe as dd
        >>> d = {'col1': [1, 2, 3, 4], 'col2': [5, 6, 7, 8]}
        >>> df = pd.DataFrame(data=d)
        >>> ddf = dd.from_pandas(df, 2)
        >>> ddf.groupby(['col1']).col2.nunique().compute()
        """
        name = self._meta.obj.name
        levels = _determine_levels(self.by)
        if isinstance(self.obj, DataFrame):
            chunk = _nunique_df_chunk
        else:
            chunk = _nunique_series_chunk
        return aca([self.obj, self.by] if not isinstance(self.by, list) else [self.obj] + self.by, chunk=chunk, aggregate=_nunique_df_aggregate, combine=_nunique_df_combine, token='series-groupby-nunique', chunk_kwargs={'levels': levels, 'name': name}, aggregate_kwargs={'levels': levels, 'name': name}, combine_kwargs={'levels': levels}, split_every=split_every, split_out=split_out, split_out_setup=split_out_on_index, sort=self.sort)

    @_deprecated_kwarg('shuffle', 'shuffle_method')
    @_aggregate_docstring(based_on='pd.core.groupby.SeriesGroupBy.aggregate')
    def aggregate(self, arg=None, split_every=None, split_out=1, shuffle_method=None, **kwargs):
        result = super().aggregate(arg=arg, split_every=split_every, split_out=split_out, shuffle_method=shuffle_method, **kwargs)
        if self._slice:
            try:
                result = result[self._slice]
            except KeyError:
                pass
        if arg is not None and (not isinstance(arg, (list, dict))) and isinstance(result, DataFrame):
            result = result[result.columns[0]]
        return result

    @_deprecated_kwarg('shuffle', 'shuffle_method')
    @_aggregate_docstring(based_on='pd.core.groupby.SeriesGroupBy.agg')
    def agg(self, arg=None, split_every=None, split_out=1, shuffle_method=None, **kwargs):
        return self.aggregate(arg=arg, split_every=split_every, split_out=split_out, shuffle_method=shuffle_method, **kwargs)

    @_deprecated_kwarg('shuffle', 'shuffle_method')
    @derived_from(pd.core.groupby.SeriesGroupBy)
    def value_counts(self, split_every=None, split_out=1, shuffle_method=None):
        return self._single_agg(func=_value_counts, token='value_counts', aggfunc=_value_counts_aggregate, split_every=split_every, split_out=split_out, shuffle_method=shuffle_method, columns=self._meta.apply(pd.Series).name)

    @_deprecated_kwarg('shuffle', 'shuffle_method')
    @derived_from(pd.core.groupby.SeriesGroupBy)
    def unique(self, split_every=None, split_out=1, shuffle_method=None):
        name = self._meta.obj.name
        return self._single_agg(func=M.unique, token='unique', aggfunc=_unique_aggregate, aggregate_kwargs={'name': name}, split_every=split_every, split_out=split_out, shuffle_method=shuffle_method)

    @_deprecated_kwarg('shuffle', 'shuffle_method')
    @derived_from(pd.core.groupby.SeriesGroupBy)
    def tail(self, n=5, split_every=None, split_out=1, shuffle_method=None):
        index_levels = len(self.by) if isinstance(self.by, list) else 1
        return self._single_agg(func=_tail_chunk, token='tail', aggfunc=_tail_aggregate, meta=M.tail(self._meta_nonempty), chunk_kwargs={'n': n}, aggregate_kwargs={'n': n, 'index_levels': index_levels}, split_every=split_every, split_out=split_out, shuffle_method=shuffle_method)

    @_deprecated_kwarg('shuffle', 'shuffle_method')
    @derived_from(pd.core.groupby.SeriesGroupBy)
    def head(self, n=5, split_every=None, split_out=1, shuffle_method=None):
        index_levels = len(self.by) if isinstance(self.by, list) else 1
        return self._single_agg(func=_head_chunk, token='head', aggfunc=_head_aggregate, meta=M.head(self._meta_nonempty), chunk_kwargs={'n': n}, aggregate_kwargs={'n': n, 'index_levels': index_levels}, split_every=split_every, split_out=split_out, shuffle_method=shuffle_method)