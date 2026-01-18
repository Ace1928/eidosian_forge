import functools
import inspect
from typing import Any, Dict, Tuple
import six
from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util import nest
def _to_tensor_or_tensor_spec(x):
    return x if isinstance(x, (tensor.Tensor, tensor.TensorSpec)) else ops.convert_to_tensor(x)