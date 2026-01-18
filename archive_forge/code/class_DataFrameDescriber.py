from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas._libs.tslibs import Timestamp
from pandas._typing import (
from pandas.util._validators import validate_percentile
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.arrays.floating import Float64Dtype
from pandas.core.reshape.concat import concat
from pandas.io.formats.format import format_percentiles
class DataFrameDescriber(NDFrameDescriberAbstract):
    """Class responsible for creating dataobj description.

    Parameters
    ----------
    obj : DataFrame
        DataFrame to be described.
    include : 'all', list-like of dtypes or None
        A white list of data types to include in the result.
    exclude : list-like of dtypes or None
        A black list of data types to omit from the result.
    """
    obj: DataFrame

    def __init__(self, obj: DataFrame, *, include: str | Sequence[str] | None, exclude: str | Sequence[str] | None) -> None:
        self.include = include
        self.exclude = exclude
        if obj.ndim == 2 and obj.columns.size == 0:
            raise ValueError('Cannot describe a DataFrame without columns')
        super().__init__(obj)

    def describe(self, percentiles: Sequence[float] | np.ndarray) -> DataFrame:
        data = self._select_data()
        ldesc: list[Series] = []
        for _, series in data.items():
            describe_func = select_describe_func(series)
            ldesc.append(describe_func(series, percentiles))
        col_names = reorder_columns(ldesc)
        d = concat([x.reindex(col_names, copy=False) for x in ldesc], axis=1, sort=False)
        d.columns = data.columns.copy()
        return d

    def _select_data(self) -> DataFrame:
        """Select columns to be described."""
        if self.include is None and self.exclude is None:
            default_include: list[npt.DTypeLike] = [np.number, 'datetime']
            data = self.obj.select_dtypes(include=default_include)
            if len(data.columns) == 0:
                data = self.obj
        elif self.include == 'all':
            if self.exclude is not None:
                msg = "exclude must be None when include is 'all'"
                raise ValueError(msg)
            data = self.obj
        else:
            data = self.obj.select_dtypes(include=self.include, exclude=self.exclude)
        return data