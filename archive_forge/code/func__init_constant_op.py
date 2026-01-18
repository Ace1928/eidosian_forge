from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import tf_export
def _init_constant_op(self, v, dtype):

    def init():
        init_constant = gen_array_ops.fill(array_ops.shape(v), self._initial_accumulator_value)
        return math_ops.cast(init_constant, dtype)
    return init