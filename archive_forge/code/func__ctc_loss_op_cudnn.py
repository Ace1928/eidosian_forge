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
def _ctc_loss_op_cudnn(labels, logits, logit_length, logits_time_major, blank_index):
    part_before = logits[:, :, :blank_index]
    part_after = logits[:, :, blank_index + 1:]
    part_blank = logits[:, :, blank_index:blank_index + 1]
    logits = array_ops.concat([part_blank, part_before, part_after], axis=2)
    labels = sparse_tensor.SparseTensor(labels.indices, array_ops.where(labels.values < blank_index, labels.values + 1, labels.values), labels.dense_shape)
    return _ctc_loss_impl(labels=labels, inputs=logits, sequence_length=logit_length, time_major=logits_time_major, use_cudnn=True)