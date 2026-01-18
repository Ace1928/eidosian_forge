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
def _get_counter_size(alg):
    if alg == random_ops_util.Algorithm.PHILOX.value:
        return PHILOX_COUNTER_SIZE
    elif alg == random_ops_util.Algorithm.THREEFRY.value:
        return THREEFRY_COUNTER_SIZE
    elif alg == random_ops_util.Algorithm.AUTO_SELECT.value:
        return PHILOX_COUNTER_SIZE
    else:
        raise ValueError(stateless_random_ops.unsupported_alg_error_msg(alg))