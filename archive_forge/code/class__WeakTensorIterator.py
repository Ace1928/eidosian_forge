from typing import Optional
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.types import core
class _WeakTensorIterator(object):
    """Iterates over the leading dim of a WeakTensor. Performs no error checks."""
    __slots__ = ['_weak_tensor', '_index', '_limit']

    def __init__(self, weak_tensor, dim0):
        self._weak_tensor = weak_tensor
        self._index = 0
        self._limit = dim0

    def __iter__(self):
        return self

    def __next__(self):
        if self._index == self._limit:
            raise StopIteration
        result = WeakTensor.from_tensor(self._weak_tensor.tensor[self._index])
        self._index += 1
        return result