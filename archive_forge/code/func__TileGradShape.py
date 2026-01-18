import numbers
import numpy as np
from tensorflow.core.config import flags
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import shape_util
from tensorflow.python.ops.gen_array_ops import *
from tensorflow.python.ops.gen_array_ops import reverse_v2 as reverse  # pylint: disable=unused-import
from tensorflow.python.types import core
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export
def _TileGradShape(op):
    """Shape function for the TileGrad op."""
    multiples_shape = op.inputs[1].get_shape().with_rank(1)
    input_shape = op.inputs[0].get_shape().with_rank(multiples_shape[0])
    multiples = tensor_util.constant_value_as_shape(op.inputs[1]).with_rank(input_shape.ndims)
    if multiples.ndims is None:
        return [tensor_shape.unknown_shape()]
    else:
        output_dims = []
        for dim, multiple in zip(input_shape.dims, multiples.dims):
            output_dims.append(dim // multiple)
        return [tensor_shape.TensorShape(output_dims)]