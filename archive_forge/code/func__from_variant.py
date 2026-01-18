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
@classmethod
def _from_variant(cls, variant, dtype, output_ragged_rank, input_ragged_rank=None, row_splits_dtype=dtypes.int64, name=None):
    """Converts a `variant` Tensor into a `RaggedTensor`.

    The input `variant` could be a scalar, meaning it encodes a single
    `RaggedTensor` with ragged_rank `output_ragged_rank`. Alternatively it could
    have an arbitrary rank, in which case each element is decoded into a
    `RaggedTensor` with ragged_rank `input_ragged_rank` and these are then
    stacked according to the input shape to output a single `RaggedTensor`
    with ragged_rank `output_ragged_rank`. If `input_ragged_rank` is not
    provided, it is inferred dynamically as `output_ragged_rank` -
    `rank(variant)`. If `input_ragged_rank` is provided, the following must be
    true: `output_ragged_rank` = `input_ragged_rank` + `rank(variant)`.

    Example:

    >>> rt = tf.ragged.constant([[0], [1, 2]])
    >>> et = rt._to_variant()
    >>> stacked_et = tf.stack([et, et])
    >>> tf.RaggedTensor._from_variant(  # scalar input.
    ...     et, dtype=tf.int32, output_ragged_rank=1).to_list()
    [[0], [1, 2]]
    >>> tf.RaggedTensor._from_variant(  # batched input.
    ...     stacked_et, dtype=tf.int32, output_ragged_rank=2).to_list()
    [[[0], [1, 2]], [[0], [1, 2]]]

    Args:
      variant: A `variant` Tensor representing an encoded (possibly
        nested-batched) `RaggedTensor`.
      dtype: The dtype of the encoded `RaggedTensor`.
      output_ragged_rank: The expected ragged rank of the output `RaggedTensor`.
      input_ragged_rank: The ragged rank of each encoded `RaggedTensor`. This is
        optional and inferred dynamically if not provided.
      row_splits_dtype: `dtype` for the RaggedTensor's `row_splits` tensor. One
        of `tf.int32` or `tf.int64`.
      name: A name prefix for the returned tensors (optional).

    Returns:
      A `RaggedTensor` of dtype `dtype` and ragged rank `output_ragged_rank`.

    Raises:
      ValueError: If the input rank is known, `input_ragged_rank` is provided
          and `output_ragged_rank` = `input_ragged_rank` + `rank(variant)` does
          not hold.
    """
    variant = ops.convert_to_tensor(variant, name='variant', dtype=dtypes.variant)
    if variant.shape.ndims is not None and input_ragged_rank is not None and (output_ragged_rank != input_ragged_rank + variant.shape.ndims):
        raise ValueError(f'Argument `output_ragged_rank` ({output_ragged_rank}) must be equal to `input_ragged_rank` + `variant.shape.ndims` ({input_ragged_rank} + {variant.shape.ndims}).')
    input_ragged_rank = -1 if input_ragged_rank is None else input_ragged_rank
    with ops.name_scope(name, 'RaggedFromVariant', [variant, dtype, input_ragged_rank, output_ragged_rank]):
        result = gen_ragged_conversion_ops.ragged_tensor_from_variant(variant, input_ragged_rank, max(output_ragged_rank, 0), dtype, row_splits_dtype, name)
        return cls.from_nested_row_splits(result.output_dense_values, result.output_nested_splits, validate=False)