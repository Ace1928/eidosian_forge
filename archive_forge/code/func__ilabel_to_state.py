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
def _ilabel_to_state(labels, num_labels, ilabel_log_probs):
    """Project ilabel log probs to state log probs."""
    num_label_states = _get_dim(labels, 1)
    blank = ilabel_log_probs[:, :, :1]
    blank = array_ops.tile(blank, [1, 1, num_label_states + 1])
    one_hot = array_ops.one_hot(labels, depth=num_labels)
    one_hot = array_ops.expand_dims(one_hot, axis=0)
    ilabel_log_probs = array_ops.expand_dims(ilabel_log_probs, axis=2)
    state_log_probs = math_ops.reduce_sum(ilabel_log_probs * one_hot, axis=3)
    state_log_probs = array_ops.concat([state_log_probs, blank], axis=2)
    return array_ops.pad(state_log_probs, [[0, 0], [0, 0], [1, 0]], constant_values=math_ops.log(0.0))