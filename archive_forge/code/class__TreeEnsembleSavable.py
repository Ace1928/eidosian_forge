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
class _TreeEnsembleSavable(saver.BaseSaverBuilder.SaveableObject):
    """SaveableObject implementation for TreeEnsemble."""

    def __init__(self, resource_handle, create_op, name):
        """Creates a _TreeEnsembleSavable object.

    Args:
      resource_handle: handle to the decision tree ensemble variable.
      create_op: the op to initialize the variable.
      name: the name to save the tree ensemble variable under.
    """
        stamp_token, serialized = gen_boosted_trees_ops.boosted_trees_serialize_ensemble(resource_handle)
        slice_spec = ''
        specs = [saver.BaseSaverBuilder.SaveSpec(stamp_token, slice_spec, name + '_stamp'), saver.BaseSaverBuilder.SaveSpec(serialized, slice_spec, name + '_serialized')]
        super(_TreeEnsembleSavable, self).__init__(resource_handle, specs, name)
        self.resource_handle = resource_handle
        self._create_op = create_op

    def restore(self, restored_tensors, unused_restored_shapes):
        """Restores the associated tree ensemble from 'restored_tensors'.

    Args:
      restored_tensors: the tensors that were loaded from a checkpoint.
      unused_restored_shapes: the shapes this object should conform to after
        restore. Not meaningful for trees.

    Returns:
      The operation that restores the state of the tree ensemble variable.
    """
        with ops.control_dependencies([self._create_op]):
            return gen_boosted_trees_ops.boosted_trees_deserialize_ensemble(self.resource_handle, stamp_token=restored_tensors[0], tree_ensemble_serialized=restored_tensors[1])