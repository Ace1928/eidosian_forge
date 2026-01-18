from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
from six.moves import range
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework.ops import internal_convert_to_tensor
from tensorflow.python.framework.ops import name_scope
from tensorflow.python.ops import gen_sdca_ops
from tensorflow.python.ops import variables as var_ops
from tensorflow.python.ops.nn import log_poisson_loss
from tensorflow.python.ops.nn import sigmoid_cross_entropy_with_logits
from tensorflow_estimator.python.estimator.canned.linear_optimizer.python.utils.sharded_mutable_dense_hashtable import _ShardedMutableDenseHashTable
def _get_partitioned_update_ops(self, v_num, num_partitions_by_var, p_assignments_by_var, gather_ids_by_var, weights, full_update, p_assignments, num_partitions):
    """Get updates for partitioned variables."""
    num_partitions = num_partitions_by_var[v_num]
    p_assignments = p_assignments_by_var[v_num]
    gather_ids = gather_ids_by_var[v_num]
    updates = tf.dynamic_partition(full_update, p_assignments, num_partitions)
    update_ops = []
    for p in range(num_partitions):
        with ops.colocate_with(weights[p]):
            result = tf.compat.v1.scatter_add(weights[p], gather_ids[p], updates[p])
        update_ops.append(result)
    return update_ops