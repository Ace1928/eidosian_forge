import collections
import copy
import math
import re
from typing import Optional
from tensorflow.core.protobuf.tpu import optimization_parameters_pb2
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2 as elc
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_system_metadata as tpu_system_metadata_lib
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.util.tf_export import tf_export
def _format_for_tpu_embedding_arbitrary_tensor_batch(self, enqueue_datas, ragged):
    """Format features for `enqueue_tpu_embedding_arbitrary_tensor_batch()`.

    Args:
      enqueue_datas: a `Dict` of `RaggedEnqueueData` objects for embedding.
      ragged: If True, extract row splits from the data rather than sample
        indices.

    Returns:
      Dict of arguments for `enqueue_tpu_embedding_arbitrary_tensor_batch()`.
    """
    kwargs = {'sample_indices_or_row_splits': [], 'embedding_indices': [], 'aggregation_weights': []}
    int_zeros = array_ops.zeros((0,), dtype=dtypes.int64)
    float_zeros = array_ops.zeros((0,), dtype=dtypes.float32)
    for table in self._table_to_features_dict:
        features = self._table_to_features_dict[table]
        for feature in features:
            enqueue_data = enqueue_datas[feature]
            if ragged:
                kwargs['sample_indices_or_row_splits'].append(enqueue_data.row_splits if enqueue_data.row_splits is not None else int_zeros)
            elif self._feature_to_config_dict[feature].max_sequence_length > 0 and enqueue_data.sample_indices is not None and (enqueue_data.sample_indices.shape[1] == 2):
                sample_indices = array_ops.pad(enqueue_data.sample_indices, paddings=[[0, 0], [0, 1]])
                kwargs['sample_indices_or_row_splits'].append(sample_indices)
            elif enqueue_data.sample_indices is None or enqueue_data.sample_indices.shape[1] == 1:
                kwargs['sample_indices_or_row_splits'].append(int_zeros)
            else:
                kwargs['sample_indices_or_row_splits'].append(enqueue_data.sample_indices)
            kwargs['aggregation_weights'].append(enqueue_data.aggregation_weights if enqueue_data.aggregation_weights is not None else float_zeros)
            kwargs['embedding_indices'].append(enqueue_data.embedding_indices)
    return kwargs