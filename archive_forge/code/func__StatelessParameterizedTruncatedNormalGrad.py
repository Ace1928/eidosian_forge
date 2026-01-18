import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import math_ops
@ops.RegisterGradient('StatelessParameterizedTruncatedNormal')
def _StatelessParameterizedTruncatedNormalGrad(op, grad):
    """Returns the gradient of a TruncatedNormal sample w.r.t. parameters.

  The gradient is computed using implicit differentiation
  (Figurnov et al., 2018).

  Args:
    op: A `StatelessParameterizedTruncatedNormal` operation. We assume that the
      inputs to the operation are `shape`, `seed`, `mean`, `stddev`, `minval`,
      and `maxval` tensors, and the output is the `sample` tensor.
    grad: The incoming gradient `dloss / dsample` of the same shape as
      `op.outputs[0]`.

  Returns:
    A list of `Tensor` with derivates with respect to each parameter.

  References:
    Implicit Reparameterization Gradients:
      [Figurnov et al., 2018]
      (http://papers.nips.cc/paper/7326-implicit-reparameterization-gradients)
      ([pdf]
      (http://papers.nips.cc/paper/7326-implicit-reparameterization-gradients.pdf))
  """
    shape = op.inputs[0]
    mean = op.inputs[2]
    stddev = op.inputs[3]
    minval = op.inputs[4]
    maxval = op.inputs[5]
    sample = op.outputs[0]
    with ops.control_dependencies([grad]):
        minval_std = (minval - mean) / stddev
        maxval_std = (maxval - mean) / stddev
        sample_std = (sample - mean) / stddev
        cdf_sample = (_Ndtr(sample_std) - _Ndtr(minval_std)) / (_Ndtr(maxval_std) - _Ndtr(minval_std))
        tiny = np.finfo(mean.dtype.as_numpy_dtype).tiny
        eps = np.finfo(mean.dtype.as_numpy_dtype).eps
        cdf_sample = clip_ops.clip_by_value(cdf_sample, tiny, 1 - eps)
        dmaxval = math_ops.exp(0.5 * (sample_std ** 2 - maxval_std ** 2) + math_ops.log(cdf_sample))
        dminval = math_ops.exp(0.5 * (sample_std ** 2 - minval_std ** 2) + math_ops.log1p(-cdf_sample))
        dmean = array_ops.ones_like(sample_std)
        dstddev = sample_std
        mean_shape = array_ops.shape(mean)
        stddev_shape = array_ops.shape(stddev)
        minval_shape = array_ops.shape(minval)
        maxval_shape = array_ops.shape(maxval)
        broadcast_shape = array_ops.broadcast_dynamic_shape(mean_shape, stddev_shape)
        broadcast_shape = array_ops.broadcast_dynamic_shape(minval_shape, broadcast_shape)
        broadcast_shape = array_ops.broadcast_dynamic_shape(maxval_shape, broadcast_shape)
        extra_dims = math_ops.range(array_ops.size(shape) - array_ops.size(broadcast_shape))
        grad_mean = math_ops.reduce_sum(grad * dmean, axis=extra_dims)
        grad_stddev = math_ops.reduce_sum(grad * dstddev, axis=extra_dims)
        grad_minval = math_ops.reduce_sum(grad * dminval, axis=extra_dims)
        grad_maxval = math_ops.reduce_sum(grad * dmaxval, axis=extra_dims)
        _, rmean = gen_array_ops.broadcast_gradient_args(broadcast_shape, mean_shape)
        _, rstddev = gen_array_ops.broadcast_gradient_args(broadcast_shape, stddev_shape)
        _, rminval = gen_array_ops.broadcast_gradient_args(broadcast_shape, minval_shape)
        _, rmaxval = gen_array_ops.broadcast_gradient_args(broadcast_shape, maxval_shape)
        grad_mean = array_ops.reshape(math_ops.reduce_sum(grad_mean, axis=rmean, keepdims=True), mean_shape)
        grad_stddev = array_ops.reshape(math_ops.reduce_sum(grad_stddev, axis=rstddev, keepdims=True), stddev_shape)
        grad_minval = array_ops.reshape(math_ops.reduce_sum(grad_minval, axis=rminval, keepdims=True), minval_shape)
        grad_maxval = array_ops.reshape(math_ops.reduce_sum(grad_maxval, axis=rmaxval, keepdims=True), maxval_shape)
        return (None, None, grad_mean, grad_stddev, grad_minval, grad_maxval)