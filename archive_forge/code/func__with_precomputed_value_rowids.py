import numpy as np
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_ragged_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import segment_id_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util.tf_export import tf_export
def _with_precomputed_value_rowids(self):
    """Returns a copy of `self` with `value_rowids` precomputed."""
    return RowPartition(row_splits=self._row_splits, row_lengths=self._row_lengths, value_rowids=self.value_rowids(), nrows=self._nrows, nvals=self._nvals, uniform_row_length=self._uniform_row_length, internal=_row_partition_factory_key)