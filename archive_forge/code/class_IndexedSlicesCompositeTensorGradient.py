import collections
import warnings
import numpy as np
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import internal
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
class IndexedSlicesCompositeTensorGradient(composite_tensor_gradient.CompositeTensorGradient):
    """CompositeTensorGradient for IndexedSlices."""

    def get_gradient_components(self, value):
        return value

    def replace_gradient_components(self, value, component_grads):
        return component_grads