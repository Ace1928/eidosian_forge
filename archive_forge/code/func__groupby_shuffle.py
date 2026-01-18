import ast
import hashlib
import re
import warnings
from collections.abc import Iterable
from typing import Hashable, List
import numpy as np
import pandas
from pandas._libs import lib
from pandas.api.types import is_scalar
from pandas.core.apply import reconstruct_func
from pandas.core.common import is_bool_indexer
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import (
from pandas.core.groupby.base import transformation_kernels
from pandas.core.indexes.api import ensure_index_from_sequences
from pandas.core.indexing import check_bool_indexer
from pandas.errors import DataError
from modin.config import CpuCount, RangePartitioning, use_range_partitioning_groupby
from modin.core.dataframe.algebra import (
from modin.core.dataframe.algebra.default2pandas.groupby import (
from modin.core.dataframe.pandas.metadata import (
from modin.core.storage_formats import BaseQueryCompiler
from modin.error_message import ErrorMessage
from modin.logging import get_logger
from modin.utils import (
from .aggregations import CorrCovBuilder
from .groupby import GroupbyReduceImpl, PivotTableImpl
from .merge import MergeImpl
from .utils import get_group_names, merge_partitioning
@_inherit_docstrings(BaseQueryCompiler.groupby_agg)
def _groupby_shuffle(self, by, agg_func, axis, groupby_kwargs, agg_args, agg_kwargs, drop=False, how='axis_wise', series_groupby=False):
    if len(self.columns) == 0 or len(self._modin_frame) == 0:
        return super().groupby_agg(by, agg_func, axis, groupby_kwargs, agg_args, agg_kwargs, how, drop)
    grouping_on_level = groupby_kwargs.get('level') is not None
    if any((isinstance(obj, pandas.Grouper) for obj in (by if isinstance(by, list) else [by]))):
        raise NotImplementedError('Grouping on a pandas.Grouper with range-partitioning groupby is not yet supported: ' + 'https://github.com/modin-project/modin/issues/5926')
    if grouping_on_level:
        external_by, internal_by, by_positions = ([], [], [])
    else:
        external_by, internal_by, by_positions = self._groupby_separate_by(by, drop)
    all_external_are_qcs = all((isinstance(obj, type(self)) for obj in external_by))
    if not all_external_are_qcs:
        raise NotImplementedError("Grouping on an external grouper with range-partitioning groupby is only supported with Series'es: " + 'https://github.com/modin-project/modin/issues/5926')
    is_transform = how == 'transform' or GroupBy.is_transformation_kernel(agg_func)
    if is_transform:
        ErrorMessage.mismatch_with_pandas(operation='range-partitioning groupby', message='the order of rows may be shuffled for the result')
    if not is_transform and groupby_kwargs.get('observed', False) in (False, lib.no_default):
        internal_dtypes = pandas.Series()
        external_dtypes = pandas.Series()
        if len(internal_by) > 0:
            internal_dtypes = self._modin_frame._dtypes.lazy_get(internal_by).get() if isinstance(self._modin_frame._dtypes, ModinDtypes) else self.dtypes[internal_by]
        if len(external_by) > 0:
            dtypes_list = []
            for obj in external_by:
                if not isinstance(obj, type(self)):
                    continue
                dtypes_list.append(obj.dtypes)
            external_dtypes = pandas.concat(dtypes_list)
        by_dtypes = pandas.concat([internal_dtypes, external_dtypes])
        add_missing_cats = any((isinstance(dtype, pandas.CategoricalDtype) for dtype in by_dtypes))
    else:
        add_missing_cats = False
    if add_missing_cats and (not groupby_kwargs.get('as_index', True)):
        raise NotImplementedError('Range-partitioning groupby is not implemented for grouping on categorical columns with ' + "the following set of parameters {'as_index': False, 'observed': False}. Change either 'as_index' " + "or 'observed' to True and try again. " + 'https://github.com/modin-project/modin/issues/5926')
    if isinstance(agg_func, dict):
        assert how == 'axis_wise', f"Only 'axis_wise' aggregation is supported with dictionary functions, got: {how}"
        subset = internal_by + list(agg_func.keys())
        subset = list(dict.fromkeys(subset))
        obj = self.getitem_column_array(subset)
    else:
        obj = self
    agg_method = (SeriesGroupByDefault if series_groupby else GroupByDefault).get_aggregation_method(how)
    original_agg_func = agg_func

    def agg_func(grp, *args, **kwargs):
        result = agg_method(grp, original_agg_func, *args, **kwargs)
        if result.ndim == 1:
            result = result.to_frame(MODIN_UNNAMED_SERIES_LABEL if result.name is None else result.name)
        return result
    result = obj._modin_frame.groupby(axis=axis, internal_by=internal_by, external_by=[obj._modin_frame if isinstance(obj, type(self)) else obj for obj in external_by], by_positions=by_positions, series_groupby=series_groupby, operator=lambda grp: agg_func(grp, *agg_args, **agg_kwargs), align_result_columns=how == 'group_wise', add_missing_cats=add_missing_cats, **groupby_kwargs)
    result_qc = self.__constructor__(result)
    if not is_transform and (not groupby_kwargs.get('as_index', True)):
        return result_qc.reset_index(drop=True)
    return result_qc