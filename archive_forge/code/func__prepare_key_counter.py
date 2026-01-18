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
def _prepare_key_counter(self, shape):
    delta = math_ops.reduce_prod(shape)
    counter_key = self.skip(delta)
    counter_size = _get_counter_size(self.algorithm)
    counter = array_ops.bitcast(counter_key[:counter_size], dtypes.uint64)
    key = array_ops.bitcast(counter_key[counter_size:counter_size + 1], dtypes.uint64)
    key = self._preprocess_key(key)
    return (key, counter)