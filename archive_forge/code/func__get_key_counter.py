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
def _get_key_counter(seed, alg):
    """Calculates the key and counter to pass to raw RNG ops.

  This function calculates the key and counter that will be passed to
  the raw RNG ops like `StatelessRandomUniformV2`. Depending on the
  input `alg`, the key and counter may be scrambled or copied from
  `seed`. If `alg` is `"auto_select"`, the key and counter will be
  determined at runtime based on device type.

  Args:
    seed: An integer tensor of shape [2]. The seed to calculate the key and
      counter from.
    alg: The RNG algorithm. See `tf.random.stateless_uniform` for an
      explanation.

  Returns:
    A pair (key, counter) suitable for V2 stateless RNG ops like
    `StatelessRandomUniformV2`.
  """
    if alg == Algorithm.AUTO_SELECT.value:
        key, counter = gen_stateless_random_ops_v2.stateless_random_get_key_counter(seed)
    elif alg == Algorithm.PHILOX.value:
        key, counter = _philox_scramble_seed(seed)
    elif alg == Algorithm.THREEFRY.value:
        key = array_ops.reshape(_uint32s_to_uint64(math_ops.cast(seed, dtypes.uint32)), [1])
        counter = array_ops.zeros([1], dtypes.uint64)
    else:
        raise ValueError(unsupported_alg_error_msg(alg))
    return (key, counter)