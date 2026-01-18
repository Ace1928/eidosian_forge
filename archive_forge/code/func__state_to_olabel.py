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
def _state_to_olabel(labels, num_labels, states):
    """Sum state log probs to ilabel log probs."""
    num_label_states = _get_dim(labels, 1) + 1
    label_states = states[:, :, 1:num_label_states]
    blank_states = states[:, :, num_label_states:]
    one_hot = array_ops.one_hot(labels - 1, depth=num_labels - 1, on_value=0.0, off_value=math_ops.log(0.0))
    one_hot = array_ops.expand_dims(one_hot, axis=0)
    label_states = array_ops.expand_dims(label_states, axis=3)
    label_olabels = math_ops.reduce_logsumexp(label_states + one_hot, axis=2)
    blank_olabels = math_ops.reduce_logsumexp(blank_states, axis=2, keepdims=True)
    return array_ops.concat([blank_olabels, label_olabels], axis=-1)