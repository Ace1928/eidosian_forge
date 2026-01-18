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
def _get_values_repr(self) -> str:
    from pandas.io.formats import format as fmt
    assert len(self) > 0
    vals = self._internal_get_values()
    fmt_values = fmt.format_array(vals, None, float_format=None, na_rep='NaN', quoting=QUOTE_NONNUMERIC)
    fmt_values = [i.strip() for i in fmt_values]
    joined = ', '.join(fmt_values)
    result = '[' + joined + ']'
    return result