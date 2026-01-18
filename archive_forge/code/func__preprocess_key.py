from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_stateful_random_ops
from tensorflow.python.ops import gen_stateless_random_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables
from tensorflow.python.trackable import autotrackable
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _preprocess_key(self, key):
    if self._distribution_strategy is None:
        return key
    with distribute_lib.enter_or_assert_strategy(self._distribution_strategy):
        replica_id = get_replica_id()
        if replica_id is not None:
            replica_id = array_ops_stack.stack([replica_id, 0], axis=0)
            replica_id = math_ops.cast(replica_id, dtypes.uint64)
            key = gen_stateless_random_ops_v2.stateless_random_uniform_full_int_v2(shape=[1], key=key, counter=replica_id, dtype=dtypes.uint64, alg=self.algorithm)
        return key