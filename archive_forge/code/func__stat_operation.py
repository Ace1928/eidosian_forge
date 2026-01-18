from __future__ import annotations
import pickle as pkl
import re
import warnings
from typing import TYPE_CHECKING, Any, Hashable, Literal, Optional, Sequence, Union
import numpy as np
import pandas
import pandas.core.generic
import pandas.core.resample
import pandas.core.window.rolling
from pandas._libs import lib
from pandas._libs.tslibs import to_offset
from pandas._typing import (
from pandas.compat import numpy as numpy_compat
from pandas.core.common import count_not_none, pipe
from pandas.core.dtypes.common import (
from pandas.core.indexes.api import ensure_index
from pandas.core.methods.describe import _refine_percentiles
from pandas.util._validators import (
from modin import pandas as pd
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger, disable_logging
from modin.pandas.accessor import CachedAccessor, ModinAPI
from modin.pandas.utils import is_scalar
from modin.utils import _inherit_docstrings, expanduser_path_arg, try_cast_to_pandas
from .utils import _doc_binary_op, is_full_grab_slice
def _stat_operation(self, op_name: str, axis: Union[int, str], skipna: bool, numeric_only: Optional[bool]=False, **kwargs):
    """
        Do common statistic reduce operations under frame.

        Parameters
        ----------
        op_name : str
            Name of method to apply.
        axis : int or str
            Axis to apply method on.
        skipna : bool
            Exclude NA/null values when computing the result.
        numeric_only : bool, default: False
            Include only float, int, boolean columns. If None, will attempt
            to use everything, then use only numeric data.
        **kwargs : dict
            Additional keyword arguments to pass to `op_name`.

        Returns
        -------
        scalar, Series or DataFrame
            `scalar` - self is Series and level is not specified.
            `Series` - self is Series and level is specified, or
                self is DataFrame and level is not specified.
            `DataFrame` - self is DataFrame and level is specified.
        """
    axis = self._get_axis_number(axis) if axis is not None else None
    validate_bool_kwarg(skipna, 'skipna', none_allowed=False)
    if op_name == 'median':
        numpy_compat.function.validate_median((), kwargs)
    elif op_name in ('sem', 'var', 'std'):
        val_kwargs = {k: v for k, v in kwargs.items() if k != 'ddof'}
        numpy_compat.function.validate_stat_ddof_func((), val_kwargs, fname=op_name)
    else:
        numpy_compat.function.validate_stat_func((), kwargs, fname=op_name)
    if not numeric_only:
        self._validate_dtypes(numeric_only=True)
    data = self._get_numeric_data(axis if axis is not None else 0) if numeric_only else self
    result_qc = getattr(data._query_compiler, op_name)(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)
    return self._reduce_dimension(result_qc) if isinstance(result_qc, type(self._query_compiler)) else result_qc