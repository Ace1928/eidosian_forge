from __future__ import annotations
from csv import QUOTE_NONNUMERIC
from functools import partial
import operator
from shutil import get_terminal_size
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import (
from pandas._libs.arrays import NDArrayBacked
from pandas.compat.numpy import function as nv
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_bool_kwarg
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.accessor import (
from pandas.core.algorithms import (
from pandas.core.arrays._mixins import (
from pandas.core.base import (
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.ops.common import unpack_zerodim_and_defer
from pandas.core.sorting import nargsort
from pandas.core.strings.object_array import ObjectStringArrayMixin
from pandas.io.formats import console
def _internal_get_values(self) -> ArrayLike:
    """
        Return the values.

        For internal compatibility with pandas formatting.

        Returns
        -------
        np.ndarray or ExtensionArray
            A numpy array or ExtensionArray of the same dtype as
            categorical.categories.dtype.
        """
    if needs_i8_conversion(self.categories.dtype):
        return self.categories.take(self._codes, fill_value=NaT)._values
    elif is_integer_dtype(self.categories.dtype) and -1 in self._codes:
        return self.categories.astype('object').take(self._codes, fill_value=np.nan)._values
    return np.array(self)