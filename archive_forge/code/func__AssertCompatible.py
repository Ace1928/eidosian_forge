import typing
from typing import Protocol
import numpy as np
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import tensor_shape
from tensorflow.python.types import core
from tensorflow.python.types import internal
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _AssertCompatible(values, dtype):
    if dtype is None:
        fn = _check_not_tensor
    else:
        try:
            fn = _TF_TO_IS_OK[dtype]
        except KeyError:
            if dtype.is_integer:
                fn = _check_int
            elif dtype.is_floating:
                fn = _check_float
            elif dtype.is_complex:
                fn = _check_complex
            elif dtype.is_quantized:
                fn = _check_quantized
            else:
                fn = _check_not_tensor
    try:
        fn(values)
    except ValueError as e:
        [mismatch] = e.args
        if dtype is None:
            raise TypeError('Expected any non-tensor type, but got a tensor instead.')
        else:
            raise TypeError(f"Expected {dtype.name}, but got {mismatch} of type '{type(mismatch).__name__}'.")