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
def _merge_precomputed_encodings(self, other, validate=True):
    """Returns a RowPartition that merges encodings from `self` and `other`.

    Requires that `self` and `other` describe the same partition.

    Args:
      other: A `RowPartition` that encodes the same partition as `self`.
      validate: If true, then add runtime checks to verify that `self` and
        `other` encode the same row partition.

    Returns:
      A `RowPartition`.
    """
    if self is other or (self._row_splits is other._row_splits and self._row_lengths is other._row_lengths and (self._value_rowids is other._value_rowids) and (self._nrows is other._nrows) and (self._nvals is other._nvals) and (self._uniform_row_length is other._uniform_row_length)):
        return self
    nrows, nrows_validated = _merge_tensors(self._nrows, other._nrows, 'nrows', validate)
    nvals, _ = _merge_tensors(self._nvals, other._nvals, 'nvals', validate)
    uniform_row_length, uniform_row_length_validated = _merge_tensors(self._uniform_row_length, other._uniform_row_length, 'uniform_row_length', validate)
    if uniform_row_length_validated and nrows_validated:
        validate = False
    row_splits, row_splits_validated = _merge_tensors(self._row_splits, other._row_splits, 'row_splits', validate)
    if row_splits_validated:
        validate = False
    row_lengths, row_lengths_validated = _merge_tensors(self._row_lengths, other._row_lengths, 'row_lengths', validate)
    if row_lengths_validated:
        validate = False
    value_rowids, value_rowids_validated = _merge_tensors(self._value_rowids, other._value_rowids, 'value_rowids', validate)
    if value_rowids_validated and nrows_validated:
        validate = False
    if row_splits is self._row_splits and row_lengths is self._row_lengths and (value_rowids is self._value_rowids) and (nrows is self._nrows) and (uniform_row_length is self._uniform_row_length):
        return self
    if row_splits is other._row_splits and row_lengths is other._row_lengths and (value_rowids is other._value_rowids) and (nrows is other._nrows) and (uniform_row_length is other._uniform_row_length):
        return other
    return RowPartition(row_splits=row_splits, row_lengths=row_lengths, value_rowids=value_rowids, nrows=nrows, uniform_row_length=uniform_row_length, nvals=nvals, internal=_row_partition_factory_key)