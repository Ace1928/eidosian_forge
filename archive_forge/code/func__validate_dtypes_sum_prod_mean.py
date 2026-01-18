from __future__ import annotations
import os
import warnings
from typing import IO, TYPE_CHECKING, Any, Hashable, Iterable, Optional, Union
import numpy as np
import pandas
from pandas._libs import lib
from pandas._typing import ArrayLike, Axis, DtypeObj, IndexKeyFunc, Scalar, Sequence
from pandas.api.types import is_integer
from pandas.core.arrays import ExtensionArray
from pandas.core.common import apply_if_callable, is_bool_indexer
from pandas.core.dtypes.common import is_dict_like, is_list_like
from pandas.core.series import _coerce_method
from pandas.io.formats.info import SeriesInfo
from pandas.util._validators import validate_bool_kwarg
from modin.config import PersistentPickle
from modin.logging import disable_logging
from modin.pandas.io import from_pandas, to_pandas
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, _inherit_docstrings
from .accessor import CachedAccessor, SparseAccessor
from .base import _ATTRS_NO_LOOKUP, BasePandasDataset
from .iterator import PartitionIterator
from .series_utils import (
from .utils import _doc_binary_op, cast_function_modin2pandas, is_scalar
def _validate_dtypes_sum_prod_mean(self, axis, numeric_only, ignore_axis=False) -> Series:
    """
        Validate data dtype for `sum`, `prod` and `mean` methods.

        Parameters
        ----------
        axis : {0, 1}
            Axis to validate over.
        numeric_only : bool
            Whether or not to allow only numeric data.
            If True and non-numeric data is found, exception
            will be raised.
        ignore_axis : bool, default: False
            Whether or not to ignore `axis` parameter.

        Returns
        -------
        Series

        Notes
        -----
        Actually returns unmodified `self` object,
        added for compatibility with Modin DataFrame.
        """
    return self