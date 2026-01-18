from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import math_ops
@ops.RegisterGradient('ScaleAndTranslate')
def _ScaleAndTranslateGrad(op, grad):
    """The derivatives for ScaleAndTranslate transformation op.

  Args:
    op: The ScaleAndTranslate op.
    grad: The tensor representing the gradient w.r.t. the output.

  Returns:
    The gradients w.r.t. the input.
  """
    grad0 = gen_image_ops.scale_and_translate_grad(grad, op.inputs[0], op.inputs[2], op.inputs[3], kernel_type=op.get_attr('kernel_type'), antialias=op.get_attr('antialias'))
    return [grad0, None, None, None]