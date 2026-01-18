import abc
from typing import Any, Iterable, Optional, Sequence, Tuple, Union
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.ops.ragged.row_partition import RowPartitionSpec
from tensorflow.python.types import core
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
class _DynamicRaggedShapeBatchEncoder(extension_type.ExtensionTypeBatchEncoder):
    """A batch encoder for DynamicRaggedShape below."""

    def batch(self, spec: 'DynamicRaggedShape.Spec', batch_size) -> 'DynamicRaggedShape.Spec':
        if spec.num_row_partitions:
            new_head = _batch_rp_spec_head(spec._row_partitions[0], batch_size)
            new_tail = [_batch_rp_spec(rp, batch_size) for rp in spec._row_partitions]
            new_rp = [new_head] + new_tail
            new_static_inner_shape = _batch_static_inner_shape(spec._static_inner_shape, batch_size)
            return DynamicRaggedShape.Spec(row_partitions=new_rp, static_inner_shape=new_static_inner_shape, dtype=spec.dtype)
        elif batch_size is None:
            if spec.inner_rank == 0:
                return DynamicRaggedShape.Spec._from_tensor_shape([None], 0, dtype=spec.dtype)
            else:
                new_head = RowPartitionSpec(uniform_row_length=spec._dimension(0), dtype=spec.dtype)
                new_static_inner_shape = _batch_static_inner_shape(spec._static_inner_shape, batch_size)
                return DynamicRaggedShape.Spec(row_partitions=[new_head], static_inner_shape=new_static_inner_shape, dtype=spec.dtype)
        else:
            return DynamicRaggedShape.Spec(row_partitions=[], static_inner_shape=_batch_tensor_shape(spec._static_inner_shape, batch_size), dtype=spec.dtype)

    def unbatch(self, spec: 'DynamicRaggedShape.Spec') -> 'DynamicRaggedShape.Spec':
        if spec.num_row_partitions:
            result = []
            head = spec._row_partitions[0]
            scale = None if head.uniform_row_length is None else head.nrows
            for rp in spec._row_partitions[1:]:
                if scale is None:
                    result.append(RowPartitionSpec(nrows=None, nvals=None, uniform_row_length=rp.uniform_row_length, dtype=spec.dtype))
                else:
                    nrows = None if rp.nrows is None else rp.nrows // scale
                    if rp.uniform_row_length is None:
                        scale = None
                        result.append(RowPartitionSpec(nrows=nrows, nvals=None, uniform_row_length=None, dtype=spec.dtype))
                    else:
                        result.append(RowPartitionSpec(nrows=nrows, nvals=rp.nvals // scale, uniform_row_length=rp.uniform_row_length, dtype=spec.dtype))
            return DynamicRaggedShape.Spec(row_partitions=result, static_inner_shape=_unbatch_static_inner_shape(spec._static_inner_shape, scale), dtype=spec.dtype)
        else:
            return DynamicRaggedShape.Spec(row_partitions=[], static_inner_shape=spec._static_inner_shape[1:], dtype=spec.dtype)

    def decode(self, spec: 'DynamicRaggedShape.Spec', encoding) -> 'DynamicRaggedShape':
        return DynamicRaggedShape.from_tensor(encoding, dtype=spec.dtype)

    def encode(self, spec: 'DynamicRaggedShape.Spec', value, minimum_rank=0) -> Union[ragged_tensor.RaggedTensor, tensor_lib.Tensor]:
        return ones(value, dtype=dtypes.bool)

    def encoding_specs(self, spec: 'DynamicRaggedShape.Spec') -> Union[ragged_tensor.RaggedTensorSpec, tensor_lib.TensorSpec]:
        if spec.rank != 0:
            ragged_rank = spec.num_row_partitions
        else:
            ragged_rank = -1
        return ragged_tensor.RaggedTensorSpec(shape=spec._to_tensor_shape(), dtype=dtypes.bool, ragged_rank=ragged_rank, row_splits_dtype=spec.dtype)