import collections
import re
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging
from tensorflow.python.util.tf_export import tf_export
def _construct_tensors_for_composite_features(features, tensor_dict):
    """Creates tensors for SparseFeatures and RaggedFeatures.

  Constructs new dict based on `tensor_dict`.

  For each key in `features` whose value is a `SparseFeature`:

    * Looks up that SparseFeature's value_key and index_keys in tensor_dict.
    * Uses those tensors to construct a single SparseTensor.
    * Stores that SparseTensor in the output dict under the same key.

  For each key in `features` whose value is a `RaggedFeature`:

    * Looks up that RaggedFeature's value_key and partition keys in tensor_dict.
    * Uses those tensors to construct a single RaggedTensor.
    * Stores that RaggedTensor in the output dict under the same key.

  For any other key in `features`:

    * Copies that key and its value from tensor_dict to the output dictionary.

  Args:
    features: A `dict` mapping feature keys to `SparseFeature` or
      `RaggedFeature` values.  Values of other types will be ignored.
    tensor_dict: A `dict` mapping feature keys to `Tensor`, `SparseTensor`, and
      `RaggedTensor` values.  Expected to contain keys of the `SparseFeature`s'
      `index_key`s and `value_key`s and mapping them to `SparseTensor`s.

  Returns:
    A `dict` mapping feature keys to `Tensor`, `SparseTensor`, and
    `RaggedTensor` values. Similar to `tensor_dict` except each `SparseFeature`
    in `features` results in a single `SparseTensor`; and each `RaggedFeature`
    in `features` results in a single `RaggedTensor`.
  """
    tensor_dict = dict(tensor_dict)
    updates = {}
    for key in sorted(features.keys()):
        feature = features[key]
        if isinstance(feature, SparseFeature):
            if isinstance(feature.index_key, str):
                sp_ids = tensor_dict[feature.index_key]
            else:
                sp_ids = [tensor_dict[index_key] for index_key in feature.index_key]
            sp_values = tensor_dict[feature.value_key]
            updates[key] = sparse_ops.sparse_merge(sp_ids, sp_values, vocab_size=feature.size, already_sorted=feature.already_sorted)
        elif isinstance(feature, RaggedFeature):
            value_key = key if feature.value_key is None else feature.value_key
            rt = tensor_dict[value_key]
            if isinstance(rt, ragged_tensor.RaggedTensor):
                if rt.ragged_rank > 1:
                    outer_splits = rt.row_splits
                    rt = rt.values
                else:
                    outer_splits = None
                for partition in reversed(feature.partitions):
                    rt = _add_batched_ragged_partition(rt, partition, tensor_dict, key, feature.validate, outer_splits)
                if outer_splits is not None:
                    rt = ragged_tensor.RaggedTensor.from_row_splits(rt, outer_splits, validate=feature.validate)
            else:
                for partition in reversed(feature.partitions):
                    rt = _add_ragged_partition(rt, partition, tensor_dict, feature.row_splits_dtype, feature.validate)
            updates[key] = rt
    tensor_dict.update(updates)
    for key in set(tensor_dict) - set(features):
        del tensor_dict[key]
    return tensor_dict