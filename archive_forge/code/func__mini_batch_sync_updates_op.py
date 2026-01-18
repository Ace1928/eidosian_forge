from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed as random_seed_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_clustering_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.embedding_ops import embedding_lookup
from tensorflow.python.ops.gen_clustering_ops import *
def _mini_batch_sync_updates_op(self, update_in_steps, cluster_centers_var, cluster_centers_updated, total_counts):
    if self._use_mini_batch and self._mini_batch_steps_per_iteration > 1:
        assert update_in_steps is not None
        with ops.colocate_with(update_in_steps, ignore_existing=True):

            def _f():
                with ops.control_dependencies([state_ops.assign(update_in_steps, self._mini_batch_steps_per_iteration - 1)]):
                    with ops.colocate_with(cluster_centers_updated, ignore_existing=True):
                        if self._distance_metric == COSINE_DISTANCE:
                            cluster_centers = nn_impl.l2_normalize(cluster_centers_updated, dim=1)
                        else:
                            cluster_centers = cluster_centers_updated
                    with ops.colocate_with(cluster_centers_var, ignore_existing=True):
                        with ops.control_dependencies([state_ops.assign(cluster_centers_var, cluster_centers)]):
                            with ops.colocate_with(None, ignore_existing=True):
                                with ops.control_dependencies([state_ops.assign(total_counts, array_ops.zeros_like(total_counts))]):
                                    return array_ops.identity(update_in_steps)
            return cond.cond(update_in_steps <= 0, _f, lambda: state_ops.assign_sub(update_in_steps, 1))
    else:
        return control_flow_ops.no_op()