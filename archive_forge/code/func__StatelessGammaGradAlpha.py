import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import math_ops
def _StatelessGammaGradAlpha(shape, alpha, sample, grad):
    """Returns gradients of a gamma sampler wrt alpha."""
    num_sample_dimensions = array_ops.shape(shape)[0] - array_ops.rank(alpha)
    alpha_broadcastable = add_leading_unit_dimensions(alpha, num_sample_dimensions)
    partial_a = gen_random_ops.random_gamma_grad(alpha_broadcastable, sample)
    return math_ops.reduce_sum(grad * partial_a, axis=math_ops.range(num_sample_dimensions))