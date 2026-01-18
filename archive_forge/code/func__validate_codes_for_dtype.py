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
@classmethod
def _validate_codes_for_dtype(cls, codes, *, dtype: CategoricalDtype) -> np.ndarray:
    if isinstance(codes, ExtensionArray) and is_integer_dtype(codes.dtype):
        if isna(codes).any():
            raise ValueError('codes cannot contain NA values')
        codes = codes.to_numpy(dtype=np.int64)
    else:
        codes = np.asarray(codes)
    if len(codes) and codes.dtype.kind not in 'iu':
        raise ValueError('codes need to be array-like integers')
    if len(codes) and (codes.max() >= len(dtype.categories) or codes.min() < -1):
        raise ValueError('codes need to be between -1 and len(categories)-1')
    return codes