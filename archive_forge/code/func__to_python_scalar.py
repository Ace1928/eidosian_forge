import math
import numpy as np
import mxnet as mx
from ..context import current_context
from ..random import uniform
from ..base import _as_list
from . import ndarray
def _to_python_scalar(inputs, type_, name):
    """Converts "inputs", possibly typed mxnet NDArray, a numpy ndarray, other python types,
        to the given type
        """
    if hasattr(inputs, 'asscalar'):
        inputs = inputs.asscalar()
    try:
        inputs = type_(inputs)
    except:
        raise ValueError('Cannot convert %s to python %s' % (name, type_.__name__))
    return inputs