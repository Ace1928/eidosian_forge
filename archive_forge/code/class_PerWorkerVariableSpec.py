import contextlib
import copy
import functools
import threading
import weakref
import numpy as np
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import values
from tensorflow.python.distribute import values_util
from tensorflow.python.distribute.coordinator import coordinator_context
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.saved_model import save_context
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import core
class PerWorkerVariableSpec(tensor.TensorSpec):

    def __init__(self, value=None, name=None):
        super().__init__(value.shape, value.dtype, name=name)
        self._value = value

    def placeholder_value(self, placeholder_context):
        placeholder = super().placeholder_value(placeholder_context)
        handle_data_util.set_handle_data(placeholder, self._value._handle_data)
        return placeholder