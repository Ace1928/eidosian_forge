import functools
import operator
import typing
import numpy as np
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import tf2
from tensorflow.python.client import session
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_ragged_conversion_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_config
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import core as core_types
from tensorflow.python.types import internal as internal_types
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
def _add_supported_value_type(cls):
    """Register the `cls` as supported value type of RaggedTenosr.

  The cls must be a subclass of CompositeTensor, and must support:
   - Spec:
     The Spec must be a `BatchableTypeSpec`
   - Properties:
     - x.shape
     - x.dtype
   - Methods:
     - x.__getitem__(idx) (method: returns a supported value type)
     - x.set_shape(shape)
   - Ops:
     - tf.shape(x) -- tf.shape(x)[0] must be a tf.Tensor.
     - tf.tile(x)
     - assert_rank_at_least(x)
     - tf.ones_like(x)
     - tf.gather(params=x, indices=Tensor)
     - tf.add(x, y)
     - tf.boolean_mask(x, ...)
     - @TODO(edloper): Complete this list

   Note: the following RaggedTensor, RaggedTensorSpec methods & ops are not
   currently supported unless `rt.values` is a RaggedTensor or a tf.Tensor:
     - rt.to_tensor()
     - rt.to_sparse_tensor()
     - rt._to_variant()
     - rt._from_variant()
     - tf.ragged.cross([rt])
     - tf.gather(params=x, indices=rt)  # rt used for indices
     - RaggedTensorSpec methods:
       - _batch
       - _unbatch
       - _to_tensor_list
       - _to_batched_tensor_list
       - _from_compatible_tensor_list

  Args:
    cls: The type to be added to supported value types.
  """
    if not issubclass(cls, composite_tensor.CompositeTensor):
        raise ValueError(f'cls ({cls}) must be a subclass of CompositeTensor.')
    if not hasattr(cls, 'shape'):
        raise ValueError('cls must support the `shape` property.')
    if not hasattr(cls, 'dtype'):
        raise ValueError('cls must support the `dtype` property.')
    global _SUPPORTED_RAGGED_VALUE_TYPES
    _SUPPORTED_RAGGED_VALUE_TYPES += (cls,)