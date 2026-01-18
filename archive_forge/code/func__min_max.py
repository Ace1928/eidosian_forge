from __future__ import annotations
from collections import abc
import numbers
import operator
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
import pandas._libs.sparse as splib
from pandas._libs.sparse import (
from pandas._libs.tslibs import NaT
from pandas.compat.numpy import function as nv
from pandas.errors import PerformanceWarning
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import (
from pandas.core.dtypes.astype import astype_array
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core import arraylike
import pandas.core.algorithms as algos
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import ExtensionArray
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.indexers import (
from pandas.core.nanops import check_below_min_count
from pandas.io.formats import printing
def _min_max(self, kind: Literal['min', 'max'], skipna: bool) -> Scalar:
    """
        Min/max of non-NA/null values

        Parameters
        ----------
        kind : {"min", "max"}
        skipna : bool

        Returns
        -------
        scalar
        """
    valid_vals = self._valid_sp_values
    has_nonnull_fill_vals = not self._null_fill_value and self.sp_index.ngaps > 0
    if len(valid_vals) > 0:
        sp_min_max = getattr(valid_vals, kind)()
        if has_nonnull_fill_vals:
            func = max if kind == 'max' else min
            return func(sp_min_max, self.fill_value)
        elif skipna:
            return sp_min_max
        elif self.sp_index.ngaps == 0:
            return sp_min_max
        else:
            return na_value_for_dtype(self.dtype.subtype, compat=False)
    elif has_nonnull_fill_vals:
        return self.fill_value
    else:
        return na_value_for_dtype(self.dtype.subtype, compat=False)