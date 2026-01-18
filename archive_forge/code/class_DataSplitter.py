from __future__ import annotations
import collections
import functools
from typing import (
import numpy as np
from pandas._libs import (
import pandas._libs.groupby as libgroupby
from pandas._typing import (
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import (
from pandas.core.frame import DataFrame
from pandas.core.groupby import grouper
from pandas.core.indexes.api import (
from pandas.core.series import Series
from pandas.core.sorting import (
class DataSplitter(Generic[NDFrameT]):

    def __init__(self, data: NDFrameT, labels: npt.NDArray[np.intp], ngroups: int, *, sort_idx: npt.NDArray[np.intp], sorted_ids: npt.NDArray[np.intp], axis: AxisInt=0) -> None:
        self.data = data
        self.labels = ensure_platform_int(labels)
        self.ngroups = ngroups
        self._slabels = sorted_ids
        self._sort_idx = sort_idx
        self.axis = axis
        assert isinstance(axis, int), axis

    def __iter__(self) -> Iterator:
        sdata = self._sorted_data
        if self.ngroups == 0:
            return
        starts, ends = lib.generate_slices(self._slabels, self.ngroups)
        for start, end in zip(starts, ends):
            yield self._chop(sdata, slice(start, end))

    @cache_readonly
    def _sorted_data(self) -> NDFrameT:
        return self.data.take(self._sort_idx, axis=self.axis)

    def _chop(self, sdata, slice_obj: slice) -> NDFrame:
        raise AbstractMethodError(self)