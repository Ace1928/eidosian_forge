import numpy as np
from tensorflow.python.compat import compat
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
@ops.RegisterGradient('Sum')
def _SumGrad(op, grad):
    """Gradient for Sum."""
    input_0_shape = op.inputs[0]._shape_tuple()
    if input_0_shape is not None:
        axes = tensor_util.constant_value(op.inputs[1])
        if axes is not None:
            rank = len(input_0_shape)
            if np.array_equal(axes, np.arange(rank)):
                if context.executing_eagerly():
                    ctx = context.context()
                    new_shape = ctx.ones_rank_cache().get(rank)
                    if new_shape is None:
                        new_shape = constant_op.constant([1] * rank, dtype=dtypes.int32)
                        ctx.ones_rank_cache().put(rank, new_shape)
                else:
                    new_shape = [1] * rank
                grad = array_ops.reshape(grad, new_shape)
                if None not in input_0_shape:
                    input_shape = constant_op.constant(input_0_shape, dtype=dtypes.int32)
                else:
                    input_shape = array_ops.shape(op.inputs[0])
                return [array_ops.tile(grad, input_shape), None]
            elif None not in input_0_shape and (not context.executing_eagerly()):
                graph = ops.get_default_graph()
                axes = tuple(axes.reshape(-1))
                try:
                    output_shape_kept_dims, tile_scaling = graph._reduced_shape_cache[input_0_shape, axes]
                except KeyError:

                    def EvaluateAsTuple(t):
                        if tensor_util.is_tf_type(t):
                            value = tensor_util.try_evaluate_constant(t)
                            assert value is not None
                        else:
                            value = t
                        return tuple(value)
                    output_shape_kept_dims = EvaluateAsTuple(math_ops.reduced_shape(input_0_shape, axes))
                    tile_scaling = EvaluateAsTuple(_safe_shape_div(input_0_shape, output_shape_kept_dims))
                    graph._reduced_shape_cache[input_0_shape, axes] = (output_shape_kept_dims, tile_scaling)
                grad = array_ops.reshape(grad, output_shape_kept_dims)
                return [array_ops.tile(grad, tile_scaling), None]
    input_shape = array_ops.shape(op.inputs[0])
    if not op.get_attr('keep_dims'):
        with ops.colocate_with(input_shape):
            output_shape_kept_dims = math_ops.reduced_shape(input_shape, op.inputs[1])
        grad = array_ops.reshape(grad, output_shape_kept_dims)
    return [array_ops.broadcast_to(grad, input_shape), None]