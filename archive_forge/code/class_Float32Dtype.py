from __future__ import annotations
from typing import ClassVar
import numpy as np
from pandas.core.dtypes.base import register_extension_dtype
from pandas.core.dtypes.common import is_float_dtype
from pandas.core.arrays.numeric import (
@register_extension_dtype
class Float32Dtype(FloatingDtype):
    type = np.float32
    name: ClassVar[str] = 'Float32'
    __doc__ = _dtype_docstring.format(dtype='float32')