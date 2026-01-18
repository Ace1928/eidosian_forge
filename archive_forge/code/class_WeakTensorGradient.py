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
class WeakTensorGradient(composite_tensor_gradient.CompositeTensorGradient):
    """CompositeTensorGradient for WeakTensor."""

    def get_gradient_components(self, weak_tensor):
        return weak_tensor.tensor

    def replace_gradient_components(self, weak_tensor, component_grads):
        return weak_tensor._type_spec._from_components([component_grads])