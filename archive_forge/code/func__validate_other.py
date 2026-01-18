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
def _validate_other(self, other, axis, dtype_check=False, compare_index=False):
    """
        Help to check validity of other in inter-df operations.

        Parameters
        ----------
        other : modin.pandas.BasePandasDataset
            Another dataset to validate against `self`.
        axis : {None, 0, 1}
            Specifies axis along which to do validation. When `1` or `None`
            is specified, validation is done along `index`, if `0` is specified
            validation is done along `columns` of `other` frame.
        dtype_check : bool, default: False
            Validates that both frames have compatible dtypes.
        compare_index : bool, default: False
            Compare Index if True.

        Returns
        -------
        modin.pandas.BasePandasDataset
            Other frame if it is determined to be valid.

        Raises
        ------
        ValueError
            If `other` is `Series` and its length is different from
            length of `self` `axis`.
        TypeError
            If any validation checks fail.
        """
    if isinstance(other, BasePandasDataset):
        return other._query_compiler
    if not is_list_like(other):
        return other
    axis = self._get_axis_number(axis) if axis is not None else 1
    result = other
    if axis == 0:
        if len(other) != len(self._query_compiler.index):
            raise ValueError(f'Unable to coerce to Series, length must be {len(self._query_compiler.index)}: ' + f'given {len(other)}')
    elif len(other) != len(self._query_compiler.columns):
        raise ValueError(f'Unable to coerce to Series, length must be {len(self._query_compiler.columns)}: ' + f'given {len(other)}')
    if hasattr(other, 'dtype'):
        other_dtypes = [other.dtype] * len(other)
    elif is_dict_like(other):
        other_dtypes = [type(other[label]) for label in self._get_axis(axis) if label in other]
    else:
        other_dtypes = [type(x) for x in other]
    if compare_index:
        if not self.index.equals(other.index):
            raise TypeError('Cannot perform operation with non-equal index')
    if dtype_check:
        self_dtypes = self._get_dtypes()
        if is_dict_like(other):
            self_dtypes = [dtype for label, dtype in zip(self._get_axis(axis), self._get_dtypes()) if label in other]
        if not all((is_numeric_dtype(self_dtype) and is_numeric_dtype(other_dtype) or (is_object_dtype(self_dtype) and is_object_dtype(other_dtype)) or (lib.is_np_dtype(self_dtype, 'mM') and lib.is_np_dtype(self_dtype, 'mM')) or is_dtype_equal(self_dtype, other_dtype) for self_dtype, other_dtype in zip(self_dtypes, other_dtypes))):
            raise TypeError('Cannot do operation with improper dtypes')
    return result