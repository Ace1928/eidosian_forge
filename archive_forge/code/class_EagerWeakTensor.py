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
class EagerWeakTensor(core.Value, WeakTensor):
    """A weakly typed Eager Tensor."""
    __name__ = 'tf.EagerWeakTensor'

    def numpy(self):
        """Copy of the contents of this EagerWeakTensor into a NumPy array or scalar."""
        if not isinstance(self.tensor, ops.EagerTensor):
            raise ValueError('WeakTensor.numpy() is only supported in eager mode.')
        return self.tensor.numpy()

    def __complex__(self):
        return self.tensor.__complex__()

    def __int__(self):
        return self.tensor.__int__()

    def __float__(self):
        return self.tensor.__float__()

    def __index__(self):
        return self.tensor.__index__()

    def __format__(self, format_spec):
        return f'{self.tensor.__format__(format_spec)} weakly typed'

    def __array__(self, dtype=None):
        return np.array(self.tensor.__array__(dtype))