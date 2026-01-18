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
class EnqueueData(collections.namedtuple('EnqueueData', ['embedding_indices', 'sample_indices', 'aggregation_weights'])):
    """Data to be enqueued through generate_enqueue_ops()."""

    def __new__(cls, embedding_indices, sample_indices=None, aggregation_weights=None):
        """Data to be enqueued through generate_enqueue_ops().

    Args:
      embedding_indices: A rank 1 Tensor, indices into the embedding tables. It
        corresponds to sp_ids.values in embedding_lookup_sparse(). Both int32
        and int64 are allowed and will be converted to int32 internally.
      sample_indices: A rank 2 Tensor specifying the training example to which
        the corresponding embedding_indices and aggregation_weights values
        belong. It corresponds to sp_ids.indices in embedding_lookup_sparse().
        If it is None, we assume each embedding_indices belongs to a different
        sample. Both int32 and int64 are allowed and will be converted to int32
        internally.
      aggregation_weights: A rank 1 Tensor containing aggregation weights. It
        corresponds to sp_weights.values in embedding_lookup_sparse(). If it is
        None, we assume all weights are 1. Both float32 and float64 are allowed
        and will be converted to float32 internally.

    Returns:
      An EnqueueData tuple.

    """
        return super().__new__(cls, embedding_indices, sample_indices, aggregation_weights)

    @staticmethod
    def from_sparse_tensor(sp_tensor, weights=None):
        return EnqueueData(sp_tensor.values, sp_tensor.indices, aggregation_weights=weights.values if weights is not None else None)