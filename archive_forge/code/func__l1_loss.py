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
def _l1_loss(self):
    """Computes the (un-normalized) l1 loss of the model."""
    with name_scope('sdca/l1_loss'):
        sums = []
        for name in ['sparse_features_weights', 'dense_features_weights']:
            for var in self._variables[name]:
                for v in self._var_to_list(var):
                    weights = internal_convert_to_tensor(v)
                    with tf.compat.v1.device(weights.device):
                        sums.append(tf.math.reduce_sum(tf.math.abs(tf.cast(weights, tf.dtypes.float64))))
        return self._symmetric_l1_regularization() * tf.math.add_n(sums)