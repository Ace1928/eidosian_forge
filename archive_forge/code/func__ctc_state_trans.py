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
def _ctc_state_trans(label_seq):
    """Computes CTC alignment model transition matrix.

  Args:
    label_seq: tensor of shape [batch_size, max_seq_length]

  Returns:
    tensor of shape [batch_size, states, states] with a state transition matrix
    computed for each sequence of the batch.
  """
    with ops.name_scope('ctc_state_trans'):
        label_seq = ops.convert_to_tensor(label_seq, name='label_seq')
        batch_size = _get_dim(label_seq, 0)
        num_labels = _get_dim(label_seq, 1)
        num_label_states = num_labels + 1
        num_states = 2 * num_label_states
        label_states = math_ops.range(num_label_states)
        blank_states = label_states + num_label_states
        start_to_label = [[1, 0]]
        blank_to_label = array_ops_stack.stack([label_states[1:], blank_states[:-1]], 1)
        label_to_blank = array_ops_stack.stack([blank_states, label_states], 1)
        indices = array_ops.concat([start_to_label, blank_to_label, label_to_blank], 0)
        values = array_ops.ones([_get_dim(indices, 0)])
        trans = array_ops.scatter_nd(indices, values, shape=[num_states, num_states])
        trans += linalg_ops.eye(num_states)
        batch_idx = array_ops.zeros_like(label_states[2:])
        indices = array_ops_stack.stack([batch_idx, label_states[2:], label_states[1:-1]], 1)
        indices = array_ops.tile(array_ops.expand_dims(indices, 0), [batch_size, 1, 1])
        batch_idx = array_ops.expand_dims(math_ops.range(batch_size), 1) * [1, 0, 0]
        indices += array_ops.expand_dims(batch_idx, 1)
        repeats = math_ops.equal(label_seq[:, :-1], label_seq[:, 1:])
        values = 1.0 - math_ops.cast(repeats, dtypes.float32)
        batched_shape = [batch_size, num_states, num_states]
        label_to_label = array_ops.scatter_nd(indices, values, batched_shape)
        return array_ops.expand_dims(trans, 0) + label_to_label