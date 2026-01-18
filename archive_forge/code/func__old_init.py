import re
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import dynamic_ragged_shape
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
@classmethod
def _old_init(cls, fields, shape, nrows, row_partitions, internal=False):
    """Private constructor -- use factory methods to create StructuredTensors.

    This constructor builds a `StructuredTensor` from the given attributes,
    performing minimal validation.

    Args:
      fields: A dictionary mapping from string to `Tensor`, `RaggedTensor`, or
        `StructuredTensor`.  (This dict is not copied, so the caller must ensure
        that it does not get mutated via leaked references.)
      shape: `tf.TensorShape` with statically known rank.
      nrows: scalar integer `tf.Tensor`, or `None` if `shape.rank==0`.
      row_partitions: tuple of `RowPartition`s, with length `shape.rank-1`.
      internal: ignored argument.

    Returns:
      a StructuredTensor.
    """
    assert isinstance(fields, dict), fields
    assert isinstance(shape, tensor_shape.TensorShape), shape
    assert nrows is None or isinstance(nrows, tensor.Tensor), nrows
    assert row_partitions is None or isinstance(row_partitions, tuple), row_partitions
    return StructuredTensor(fields=fields, ragged_shape=_dynamic_ragged_shape_init(fields, shape, nrows, row_partitions))