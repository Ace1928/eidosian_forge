import math
from tensorflow.python.feature_column import feature_column as fc
from tensorflow.python.feature_column import feature_column_lib as fc_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_function
from tensorflow.python.tpu import tpu_replication
class _TPUEmbeddingColumn(_TPUBaseEmbeddingColumn, fc._EmbeddingColumn):
    """Core Embedding Column."""

    def __new__(cls, categorical_column, dimension, combiner='mean', layer_creator=None, ckpt_to_load_from=None, tensor_name_in_ckpt=None, max_norm=None, trainable=True, max_sequence_length=0, learning_rate_fn=None, use_safe_embedding_lookup=True, bypass_scope_validation=False):
        del bypass_scope_validation
        return fc._EmbeddingColumn.__new__(cls, categorical_column, dimension, combiner=combiner, layer_creator=layer_creator, ckpt_to_load_from=ckpt_to_load_from, tensor_name_in_ckpt=tensor_name_in_ckpt, max_norm=max_norm, trainable=trainable, use_safe_embedding_lookup=use_safe_embedding_lookup)

    def __init__(self, categorical_column, dimension, combiner='mean', layer_creator=None, ckpt_to_load_from=None, tensor_name_in_ckpt=None, max_norm=None, trainable=True, max_sequence_length=0, learning_rate_fn=None, use_safe_embedding_lookup=True, bypass_scope_validation=False):
        _TPUBaseEmbeddingColumn.__init__(self, categorical_column, max_sequence_length=max_sequence_length, learning_rate_fn=learning_rate_fn)
        self._key = None
        self._bypass_scope_validation = bypass_scope_validation

    def get_combiner(self):
        return self.combiner

    def get_embedding_table_size(self):
        """Returns num_ids and width."""
        return (self.categorical_column._num_buckets, self.dimension)

    def get_feature_key_name(self):
        """get_feature_key_name."""
        if self.is_categorical_column_weighted():
            return self.categorical_column.categorical_column.name
        return self.categorical_column.name

    def get_weight_key_name(self):
        """get_weight_key_name."""
        if self.is_categorical_column_weighted():
            return self.categorical_column.weight_feature_key
        return None

    def get_embedding_var_name(self):
        """get_embedding_var_name."""
        return self.categorical_column.name

    def get_initializer(self):
        return self._tpu_initializer

    def is_categorical_column_weighted(self):
        """Check if the categorical column of the embedding column is weighted."""
        if isinstance(self.categorical_column, (fc._WeightedCategoricalColumn, fc_lib.WeightedCategoricalColumn)):
            return True
        return False

    def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
        if tpu.under_tpu_inference_context():

            def host_computation():
                return fc._EmbeddingColumn._get_dense_tensor(self, inputs, weight_collections, trainable)
            return tpu_replication.outside_compilation(host_computation)
        if _is_running_on_cpu():
            return fc._EmbeddingColumn._get_dense_tensor(self, inputs, weight_collections, trainable)
        tensor = inputs.get(self.get_feature_key_name())
        _record_variable_scope_and_name(self.get_embedding_var_name(), 'embedding_weights', bypass_scope_validation=self._bypass_scope_validation)
        return tensor

    def _get_sequence_dense_tensor(self, inputs, weight_collections=None, trainable=None):
        if tpu.under_tpu_inference_context():

            def host_computation():
                return fc._EmbeddingColumn._get_sequence_dense_tensor(self, inputs, weight_collections, trainable)
            return tpu_replication.outside_compilation(host_computation)
        if _is_running_on_cpu():
            return fc._EmbeddingColumn._get_sequence_dense_tensor(self, inputs, weight_collections, trainable)
        tensor = inputs.get(self.get_feature_key_name())
        tensor_lengths = inputs.get(self.get_sequence_length_feature_key_name())
        tensor_lengths = array_ops.squeeze(tensor_lengths, -1)
        _record_variable_scope_and_name(self.get_embedding_var_name(), 'embedding_weights', bypass_scope_validation=self._bypass_scope_validation)
        return fc._SequenceDenseColumn.TensorSequenceLengthPair(dense_tensor=tensor, sequence_length=tensor_lengths)