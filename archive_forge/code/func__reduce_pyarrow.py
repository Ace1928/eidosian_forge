from __future__ import annotations
import functools
import operator
import re
import textwrap
from typing import (
import unicodedata
import numpy as np
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas.compat import (
from pandas.util._decorators import doc
from pandas.util._validators import validate_fillna_kwargs
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.missing import isna
from pandas.core import (
from pandas.core.algorithms import map_array
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._arrow_string_mixins import ArrowStringArrayMixin
from pandas.core.arrays._utils import to_numpy_dtype_inference
from pandas.core.arrays.base import (
from pandas.core.arrays.masked import BaseMaskedArray
from pandas.core.arrays.string_ import StringDtype
import pandas.core.common as com
from pandas.core.indexers import (
from pandas.core.strings.base import BaseStringArrayMethods
from pandas.io._util import _arrow_dtype_mapping
from pandas.tseries.frequencies import to_offset
def _reduce_pyarrow(self, name: str, *, skipna: bool=True, **kwargs) -> pa.Scalar:
    """
        Return a pyarrow scalar result of performing the reduction operation.

        Parameters
        ----------
        name : str
            Name of the function, supported values are:
            { any, all, min, max, sum, mean, median, prod,
            std, var, sem, kurt, skew }.
        skipna : bool, default True
            If True, skip NaN values.
        **kwargs
            Additional keyword arguments passed to the reduction function.
            Currently, `ddof` is the only supported kwarg.

        Returns
        -------
        pyarrow scalar

        Raises
        ------
        TypeError : subclass does not define reductions
        """
    pa_type = self._pa_array.type
    data_to_reduce = self._pa_array
    cast_kwargs = {} if pa_version_under13p0 else {'safe': False}
    if name in ['any', 'all'] and (pa.types.is_integer(pa_type) or pa.types.is_floating(pa_type) or pa.types.is_duration(pa_type) or pa.types.is_decimal(pa_type)):
        if pa.types.is_duration(pa_type):
            data_to_cmp = self._pa_array.cast(pa.int64())
        else:
            data_to_cmp = self._pa_array
        not_eq = pc.not_equal(data_to_cmp, 0)
        data_to_reduce = not_eq
    elif name in ['min', 'max', 'sum'] and pa.types.is_duration(pa_type):
        data_to_reduce = self._pa_array.cast(pa.int64())
    elif name in ['median', 'mean', 'std', 'sem'] and pa.types.is_temporal(pa_type):
        nbits = pa_type.bit_width
        if nbits == 32:
            data_to_reduce = self._pa_array.cast(pa.int32())
        else:
            data_to_reduce = self._pa_array.cast(pa.int64())
    if name == 'sem':

        def pyarrow_meth(data, skip_nulls, **kwargs):
            numerator = pc.stddev(data, skip_nulls=skip_nulls, **kwargs)
            denominator = pc.sqrt_checked(pc.count(self._pa_array))
            return pc.divide_checked(numerator, denominator)
    else:
        pyarrow_name = {'median': 'quantile', 'prod': 'product', 'std': 'stddev', 'var': 'variance'}.get(name, name)
        pyarrow_meth = getattr(pc, pyarrow_name, None)
        if pyarrow_meth is None:
            return super()._reduce(name, skipna=skipna, **kwargs)
    if name in ['any', 'all'] and 'min_count' not in kwargs:
        kwargs['min_count'] = 0
    elif name == 'median':
        kwargs['q'] = 0.5
    try:
        result = pyarrow_meth(data_to_reduce, skip_nulls=skipna, **kwargs)
    except (AttributeError, NotImplementedError, TypeError) as err:
        msg = f"'{type(self).__name__}' with dtype {self.dtype} does not support reduction '{name}' with pyarrow version {pa.__version__}. '{name}' may be supported by upgrading pyarrow."
        raise TypeError(msg) from err
    if name == 'median':
        result = result[0]
    if pc.is_null(result).as_py():
        return result
    if name in ['min', 'max', 'sum'] and pa.types.is_duration(pa_type):
        result = result.cast(pa_type)
    if name in ['median', 'mean'] and pa.types.is_temporal(pa_type):
        if not pa_version_under13p0:
            nbits = pa_type.bit_width
            if nbits == 32:
                result = result.cast(pa.int32(), **cast_kwargs)
            else:
                result = result.cast(pa.int64(), **cast_kwargs)
        result = result.cast(pa_type)
    if name in ['std', 'sem'] and pa.types.is_temporal(pa_type):
        result = result.cast(pa.int64(), **cast_kwargs)
        if pa.types.is_duration(pa_type):
            result = result.cast(pa_type)
        elif pa.types.is_time(pa_type):
            unit = get_unit_from_pa_dtype(pa_type)
            result = result.cast(pa.duration(unit))
        elif pa.types.is_date(pa_type):
            result = result.cast(pa.duration('s'))
        else:
            result = result.cast(pa.duration(pa_type.unit))
    return result