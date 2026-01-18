from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import math_ops
@ops.RegisterGradient('ResizeNearestNeighbor')
def _ResizeNearestNeighborGrad(op, grad):
    """The derivatives for nearest neighbor resizing.

  Args:
    op: The ResizeNearestNeighbor op.
    grad: The tensor representing the gradient w.r.t. the output.

  Returns:
    The gradients w.r.t. the input and the output.
  """
    image = op.inputs[0]
    if image.get_shape()[1:3].is_fully_defined():
        image_shape = image.get_shape()[1:3]
    else:
        image_shape = array_ops.shape(image)[1:3]
    grads = gen_image_ops.resize_nearest_neighbor_grad(grad, image_shape, align_corners=op.get_attr('align_corners'), half_pixel_centers=op.get_attr('half_pixel_centers'))
    return [grads, None]