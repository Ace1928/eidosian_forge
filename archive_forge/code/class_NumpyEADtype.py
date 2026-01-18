from __future__ import annotations
from datetime import (
from decimal import Decimal
import re
from typing import (
import warnings
import numpy as np
import pytz
from pandas._libs import (
from pandas._libs.interval import Interval
from pandas._libs.properties import cache_readonly
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import (
from pandas._libs.tslibs.offsets import BDay
from pandas.compat import pa_version_under10p1
from pandas.errors import PerformanceWarning
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import (
from pandas.util import capitalize_first_letter
class NumpyEADtype(ExtensionDtype):
    """
    A Pandas ExtensionDtype for NumPy dtypes.

    This is mostly for internal compatibility, and is not especially
    useful on its own.

    Parameters
    ----------
    dtype : object
        Object to be converted to a NumPy data type object.

    See Also
    --------
    numpy.dtype
    """
    _metadata = ('_dtype',)
    _supports_2d = False
    _can_fast_transpose = False

    def __init__(self, dtype: npt.DTypeLike | NumpyEADtype | None) -> None:
        if isinstance(dtype, NumpyEADtype):
            dtype = dtype.numpy_dtype
        self._dtype = np.dtype(dtype)

    def __repr__(self) -> str:
        return f'NumpyEADtype({repr(self.name)})'

    @property
    def numpy_dtype(self) -> np.dtype:
        """
        The NumPy dtype this NumpyEADtype wraps.
        """
        return self._dtype

    @property
    def name(self) -> str:
        """
        A bit-width name for this data-type.
        """
        return self._dtype.name

    @property
    def type(self) -> type[np.generic]:
        """
        The type object used to instantiate a scalar of this NumPy data-type.
        """
        return self._dtype.type

    @property
    def _is_numeric(self) -> bool:
        return self.kind in set('biufc')

    @property
    def _is_boolean(self) -> bool:
        return self.kind == 'b'

    @classmethod
    def construct_from_string(cls, string: str) -> NumpyEADtype:
        try:
            dtype = np.dtype(string)
        except TypeError as err:
            if not isinstance(string, str):
                msg = f"'construct_from_string' expects a string, got {type(string)}"
            else:
                msg = f"Cannot construct a 'NumpyEADtype' from '{string}'"
            raise TypeError(msg) from err
        return cls(dtype)

    @classmethod
    def construct_array_type(cls) -> type_t[NumpyExtensionArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        from pandas.core.arrays import NumpyExtensionArray
        return NumpyExtensionArray

    @property
    def kind(self) -> str:
        """
        A character code (one of 'biufcmMOSUV') identifying the general kind of data.
        """
        return self._dtype.kind

    @property
    def itemsize(self) -> int:
        """
        The element size of this data-type object.
        """
        return self._dtype.itemsize