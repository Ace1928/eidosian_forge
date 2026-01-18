from tensorflow.python.distribute import central_storage_strategy
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util as ds_reduce_util
from tensorflow.python.ops import clip_ops
from tensorflow.python.platform import tf_logging as logging
def _all_reduce_sum_fn(distribution, grads_and_vars):
    return distribution.extended.batch_reduce_to(ds_reduce_util.ReduceOp.SUM, grads_and_vars)