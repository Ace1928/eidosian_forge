from __future__ import annotations
from collections.abc import Hashable, Iterable, Sequence
from typing import TYPE_CHECKING, Generic, Literal, cast
import numpy as np
from numpy.typing import ArrayLike
from xarray.core import duck_array_ops, utils
from xarray.core.alignment import align, broadcast
from xarray.core.computation import apply_ufunc, dot
from xarray.core.types import Dims, T_DataArray, T_Xarray
from xarray.namedarray.utils import is_duck_dask_array
from xarray.util.deprecation_helpers import _deprecate_positional_args
def _inject_docstring(cls, cls_name):
    cls.sum_of_weights.__doc__ = _SUM_OF_WEIGHTS_DOCSTRING.format(cls=cls_name)
    cls.sum.__doc__ = _WEIGHTED_REDUCE_DOCSTRING_TEMPLATE.format(cls=cls_name, fcn='sum', on_zero='0')
    cls.mean.__doc__ = _WEIGHTED_REDUCE_DOCSTRING_TEMPLATE.format(cls=cls_name, fcn='mean', on_zero='NaN')
    cls.sum_of_squares.__doc__ = _WEIGHTED_REDUCE_DOCSTRING_TEMPLATE.format(cls=cls_name, fcn='sum_of_squares', on_zero='0')
    cls.var.__doc__ = _WEIGHTED_REDUCE_DOCSTRING_TEMPLATE.format(cls=cls_name, fcn='var', on_zero='NaN')
    cls.std.__doc__ = _WEIGHTED_REDUCE_DOCSTRING_TEMPLATE.format(cls=cls_name, fcn='std', on_zero='NaN')
    cls.quantile.__doc__ = _WEIGHTED_QUANTILE_DOCSTRING_TEMPLATE.format(cls=cls_name)