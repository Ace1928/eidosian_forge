from typing import Any, Dict, Iterable, Optional, Text, Union
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.tpu import tpu_embedding_base
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _embedding_lookup_for_sparse_tensor(self, inp: sparse_tensor.SparseTensor, weight: Optional[sparse_tensor.SparseTensor], table: tf_variables.Variable, feature: tpu_embedding_v2_utils.FeatureConfig) -> tensor.Tensor:
    """Embedding lookup for sparse tensor based on its feature config.

    Args:
      inp: a single SparseTensor input.
      weight: None or SparseTensor which has the same shape of the input.
      table: a table variable.
      feature: a feature config.

    Returns:
      Embedding lookup result.
    """

    def sparse_to_dense_computation(inp, weight):
        if weight is None:
            weight = sparse_tensor.SparseTensor(inp.indices, array_ops.ones_like(inp.values, dtype=dtypes.float32), dense_shape=inp.dense_shape)
        inp = sparse_ops.sparse_tensor_to_dense(inp)
        weight = sparse_ops.sparse_tensor_to_dense(weight)
        return (inp, weight)
    inp, weight = tpu_replication.outside_compilation(sparse_to_dense_computation, inp=inp, weight=weight)
    embeddings = embedding_ops.embedding_lookup_v2(table, inp)
    weight = array_ops.expand_dims(weight, -1)
    embeddings *= weight
    if not feature.output_shape and feature.max_sequence_length > 0:
        embeddings = self._pad_or_truncate_with_sequence_length(embeddings, feature.max_sequence_length)
    else:
        embeddings = self._apply_combiner_to_embeddings(embeddings, weight, feature.table.combiner)
    return embeddings