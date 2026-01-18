import enum
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import gen_stateless_random_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export
def _philox_scramble_seed(seed):
    """Determines the key and counter for Philox PRNG with the given seed.

  Args:
    seed: An integer tensor of shape [2]. The seed to calculate the key and
      counter from.

  Returns:
    A pair (key, counter) suitable for V2 stateless RNG ops like
    `StatelessRandomUniformV2`.
  """
    key = constant_op.constant([163851598941452064], dtypes.uint64)
    counter = math_ops.cast(seed, dtypes.uint64)
    mix = gen_stateless_random_ops_v2.stateless_random_uniform_full_int_v2([4], key=key, counter=counter, dtype=dtypes.uint32, alg=Algorithm.PHILOX.value)
    key = array_ops.reshape(_uint32s_to_uint64(mix[:2]), [1])
    counter = array_ops_stack.stack([0, _uint32s_to_uint64(mix[2:])], axis=0)
    return (key, counter)