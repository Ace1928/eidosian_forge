import abc
import collections
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.trackable import base as trackable
def _add_state_variable(self, name, shape, dtype, initializer=None, partitioner=None, use_resource=None, **kwargs):
    """Add a variable that can hold state which is updated during adapt().

    Args:
      name: Variable name.
      shape: Variable shape. Defaults to scalar if unspecified.
      dtype: The type of the variable. Defaults to `self.dtype` or `float32`.
      initializer: initializer instance (callable).
      partitioner: Partitioner to be passed to the `Trackable` API.
      use_resource: Whether to use `ResourceVariable`
      **kwargs: Additional keyword arguments. Accepted values are `getter` and
        `collections`.

    Returns:
      The created variable.
    """
    weight = self.add_weight(name=name, shape=shape, dtype=dtype, initializer=initializer, regularizer=None, trainable=False, constraint=None, partitioner=partitioner, use_resource=use_resource, **kwargs)
    self.state_variables[name] = weight
    return weight