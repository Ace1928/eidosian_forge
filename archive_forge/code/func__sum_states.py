import uuid
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_ctc_ops
from tensorflow.python.ops import inplace_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.nn_grad import _BroadcastMul
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _sum_states(idx, states):
    """Take logsumexp for each unique state out of all label states.

  Args:
    idx: tensor of shape [batch, label_length] For each sequence, indices into a
      set of unique labels as computed by calling unique.
    states: tensor of shape [frames, batch, label_length] Log probabilities for
      each label state.

  Returns:
    tensor of shape [frames, batch_size, label_length], log probabilities summed
      for each unique label of the sequence.
  """
    with ops.name_scope('sum_states'):
        idx = ops.convert_to_tensor(idx, name='idx')
        num_states = _get_dim(states, 2)
        states = array_ops.expand_dims(states, axis=2)
        one_hot = array_ops.one_hot(idx, depth=num_states, on_value=0.0, off_value=math_ops.log(0.0), axis=1)
        return math_ops.reduce_logsumexp(states + one_hot, axis=-1)