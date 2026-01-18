from __future__ import annotations
import operator
from enum import IntEnum
from ._creation_functions import asarray
from ._dtypes import (
from typing import TYPE_CHECKING, Optional, Tuple, Union, Any, SupportsIndex
import types
import cupy as np
from cupy.cuda import Device as _Device
from cupy.cuda import stream as stream_module
from cupy_backends.cuda.api import runtime
from cupy import array_api
def _promote_scalar(self, scalar: Union[bool, int, float]) -> Array:
    """
        Returns a promoted version of a Python scalar appropriate for use with
        operations on self.

        This may raise an OverflowError in cases where the scalar is an
        integer that is too large to fit in a NumPy integer dtype, or
        TypeError when the scalar type is incompatible with the dtype of self.
        """
    if isinstance(scalar, bool):
        if self.dtype not in _boolean_dtypes:
            raise TypeError('Python bool scalars can only be promoted with bool arrays')
    elif isinstance(scalar, int):
        if self.dtype in _boolean_dtypes:
            raise TypeError('Python int scalars cannot be promoted with bool arrays')
    elif isinstance(scalar, float):
        if self.dtype not in _floating_dtypes:
            raise TypeError('Python float scalars can only be promoted with floating-point arrays.')
    else:
        raise TypeError("'scalar' must be a Python scalar")
    return Array._new(np.array(scalar, self.dtype))