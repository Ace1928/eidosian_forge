from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_boosted_trees_ops
from tensorflow.python.ops import resources
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_aggregate_stats
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_bucketize
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_calculate_best_feature_split as calculate_best_feature_split
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_calculate_best_feature_split_v2 as calculate_best_feature_split_v2
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_calculate_best_gains_per_feature as calculate_best_gains_per_feature
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_center_bias as center_bias
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_create_quantile_stream_resource as create_quantile_stream_resource
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_example_debug_outputs as example_debug_outputs
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_make_quantile_summaries as make_quantile_summaries
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_make_stats_summary as make_stats_summary
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_predict as predict
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_quantile_stream_resource_add_summaries as quantile_add_summaries
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_quantile_stream_resource_deserialize as quantile_resource_deserialize
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_quantile_stream_resource_flush as quantile_flush
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_quantile_stream_resource_get_bucket_boundaries as get_bucket_boundaries
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_quantile_stream_resource_handle_op as quantile_resource_handle_op
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_sparse_aggregate_stats
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_sparse_calculate_best_feature_split as sparse_calculate_best_feature_split
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_training_predict as training_predict
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_update_ensemble as update_ensemble
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_update_ensemble_v2 as update_ensemble_v2
from tensorflow.python.ops.gen_boosted_trees_ops import is_boosted_trees_quantile_stream_resource_initialized as is_quantile_resource_initialized
from tensorflow.python.training import saver
class TreeEnsemble:
    """Creates TreeEnsemble resource."""

    def __init__(self, name, stamp_token=0, is_local=False, serialized_proto=''):
        self._stamp_token = stamp_token
        self._serialized_proto = serialized_proto
        self._is_local = is_local
        with ops.name_scope(name, 'TreeEnsemble') as name:
            self._name = name
            self.resource_handle = self._create_resource()
            self._init_op = self._initialize()
            is_initialized_op = self.is_initialized()
            if not is_local:
                ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, _TreeEnsembleSavable(self.resource_handle, self.initializer, self.resource_handle.name))
            resources.register_resource(self.resource_handle, self.initializer, is_initialized_op, is_shared=not is_local)

    def _create_resource(self):
        return gen_boosted_trees_ops.boosted_trees_ensemble_resource_handle_op(container='', shared_name=self._name, name=self._name)

    def _initialize(self):
        return gen_boosted_trees_ops.boosted_trees_create_ensemble(self.resource_handle, self._stamp_token, tree_ensemble_serialized=self._serialized_proto)

    @property
    def initializer(self):
        if self._init_op is None:
            self._init_op = self._initialize()
        return self._init_op

    def is_initialized(self):
        return gen_boosted_trees_ops.is_boosted_trees_ensemble_initialized(self.resource_handle)

    def _serialize_to_tensors(self):
        raise NotImplementedError('When the need arises, TF2 compatibility can be added by implementing this method, along with _restore_from_tensors below.')

    def _restore_from_tensors(self, restored_tensors):
        raise NotImplementedError('When the need arises, TF2 compatibility can be added by implementing this method, along with _serialize_to_tensors above.')

    def get_stamp_token(self):
        """Returns the current stamp token of the resource."""
        stamp_token, _, _, _, _ = gen_boosted_trees_ops.boosted_trees_get_ensemble_states(self.resource_handle)
        return stamp_token

    def get_states(self):
        """Returns states of the tree ensemble.

    Returns:
      stamp_token, num_trees, num_finalized_trees, num_attempted_layers and
      range of the nodes in the latest layer.
    """
        stamp_token, num_trees, num_finalized_trees, num_attempted_layers, nodes_range = gen_boosted_trees_ops.boosted_trees_get_ensemble_states(self.resource_handle)
        return (array_ops.identity(stamp_token, name='stamp_token'), array_ops.identity(num_trees, name='num_trees'), array_ops.identity(num_finalized_trees, name='num_finalized_trees'), array_ops.identity(num_attempted_layers, name='num_attempted_layers'), array_ops.identity(nodes_range, name='last_layer_nodes_range'))

    def serialize(self):
        """Serializes the ensemble into proto and returns the serialized proto.

    Returns:
      stamp_token: int64 scalar Tensor to denote the stamp of the resource.
      serialized_proto: string scalar Tensor of the serialized proto.
    """
        return gen_boosted_trees_ops.boosted_trees_serialize_ensemble(self.resource_handle)

    def deserialize(self, stamp_token, serialized_proto):
        """Deserialize the input proto and resets the ensemble from it.

    Args:
      stamp_token: int64 scalar Tensor to denote the stamp of the resource.
      serialized_proto: string scalar Tensor of the serialized proto.

    Returns:
      Operation (for dependencies).
    """
        return gen_boosted_trees_ops.boosted_trees_deserialize_ensemble(self.resource_handle, stamp_token, serialized_proto)