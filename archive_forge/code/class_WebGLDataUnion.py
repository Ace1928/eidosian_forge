from collections import namedtuple
from collections.abc import Sequence
import numbers
import math
import re
import warnings
from traitlets import (
from ipywidgets import widget_serialization
from ipydatawidgets import DataUnion, NDArrayWidget, shape_constraints
import numpy as np
class WebGLDataUnion(DataUnion):
    """A trait that accepts either a numpy array, or an NDArrayWidget reference.

    Also constrains the use of 64-bit arrays, as this is not supported by WebGL.
    """

    def validate(self, obj, value):
        was_original_array = isinstance(value, np.ndarray)
        value = super(WebGLDataUnion, self).validate(obj, value)
        array = value.array if isinstance(value, NDArrayWidget) else value
        dtype_str = str(array.dtype) if array is not Undefined else ''
        if dtype_str == 'float64' or dtype_str.endswith('int64'):
            if isinstance(value, NDArrayWidget):
                raise TraitError('Cannot use a %s data widget as a WebGL source.' % (dtype_str,))
            else:
                if was_original_array:
                    warnings.warn('64-bit data types not supported for WebGL data, casting to 32-bit.')
                value = value.astype(dtype_str.replace('64', '32'))
        return value