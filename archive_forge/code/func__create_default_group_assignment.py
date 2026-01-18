from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_tpu_ops
from tensorflow.python.ops.gen_tpu_ops import *
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_function
from tensorflow.python.util.tf_export import tf_export
def _create_default_group_assignment():
    num_shards = tpu_function.get_tpu_context().number_of_shards
    if num_shards is None:
        logging.warning('cross_replica_sum should be used within a tpu_shard_context, but got unset number_of_shards. Assuming 1.')
        num_shards = 1
    group_assignment = [list(range(num_shards))]
    return group_assignment