import functools
import itertools
import operator
from tensorflow.python.eager import backprop
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
@ops.RegisterGradient('SoftmaxCrossEntropyWithLogits')
def _SoftmaxCrossEntropyWithLogitsGrad(op, grad_loss, grad_grad):
    """Gradient function for SoftmaxCrossEntropyWithLogits."""
    softmax_grad = op.outputs[1]
    grad = _BroadcastMul(grad_loss, softmax_grad)
    logits = op.inputs[0]
    if grad_grad is not None and (not getattr(grad_grad, '_is_zeros_tensor', False)):
        softmax = nn_ops.softmax(logits)
        grad += (grad_grad - array_ops.squeeze(math_ops.matmul(array_ops.expand_dims(grad_grad, 1), array_ops.expand_dims(softmax, 2)), axis=1)) * softmax
    return (grad, _BroadcastMul(grad_loss, -nn_ops.log_softmax(logits)))