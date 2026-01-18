from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import math_ops
@ops.RegisterGradient('RGBToHSV')
def _RGBToHSVGrad(op, grad):
    """The gradients for `rgb_to_hsv` operation.

  This function is a piecewise continuous function as defined here:
  https://en.wikipedia.org/wiki/HSL_and_HSV#From_RGB
  We perform the multivariate derivative and compute all partial derivatives
  separately before adding them in the end. Formulas are given before each
  partial derivative calculation.

  Args:
    op: The `rgb_to_hsv` `Operation` that we are differentiating.
    grad: Gradient with respect to the output of the `rgb_to_hsv` op.

  Returns:
    Gradients with respect to the input of `rgb_to_hsv`.
  """
    reds = op.inputs[0][..., 0]
    greens = op.inputs[0][..., 1]
    blues = op.inputs[0][..., 2]
    saturation = op.outputs[0][..., 1]
    value = op.outputs[0][..., 2]
    dtype = op.inputs[0].dtype
    red_biggest = math_ops.cast((reds >= blues) & (reds >= greens), dtype)
    green_biggest = math_ops.cast((greens > reds) & (greens >= blues), dtype)
    blue_biggest = math_ops.cast((blues > reds) & (blues > greens), dtype)
    red_smallest = math_ops.cast((reds < blues) & (reds < greens), dtype)
    green_smallest = math_ops.cast((greens <= reds) & (greens < blues), dtype)
    blue_smallest = math_ops.cast((blues <= reds) & (blues <= greens), dtype)
    dv_dr = red_biggest
    dv_dg = green_biggest
    dv_db = blue_biggest
    ds_dr = math_ops.cast(reds > 0, dtype) * math_ops.add(red_biggest * math_ops.add(green_smallest * greens, blue_smallest * blues) * _CustomReciprocal(math_ops.square(reds)), red_smallest * -1 * _CustomReciprocal(green_biggest * greens + blue_biggest * blues))
    ds_dg = math_ops.cast(greens > 0, dtype) * math_ops.add(green_biggest * math_ops.add(red_smallest * reds, blue_smallest * blues) * _CustomReciprocal(math_ops.square(greens)), green_smallest * -1 * _CustomReciprocal(red_biggest * reds + blue_biggest * blues))
    ds_db = math_ops.cast(blues > 0, dtype) * math_ops.add(blue_biggest * math_ops.add(green_smallest * greens, red_smallest * reds) * _CustomReciprocal(math_ops.square(blues)), blue_smallest * -1 * _CustomReciprocal(green_biggest * greens + red_biggest * reds))
    dh_dr_1 = 60 * (math_ops.cast(reds > 0, dtype) * red_biggest * -1 * (greens - blues) * _CustomReciprocal(math_ops.square(saturation)) * _CustomReciprocal(math_ops.square(value)))
    dh_dr_2 = 60 * (math_ops.cast(greens > 0, dtype) * green_biggest * red_smallest * (blues - greens) * _CustomReciprocal(math_ops.square(reds - greens)))
    dh_dr_3 = 60 * (math_ops.cast(greens > 0, dtype) * green_biggest * blue_smallest * -1 * _CustomReciprocal(greens - blues))
    dh_dr_4 = 60 * (math_ops.cast(blues > 0, dtype) * blue_biggest * red_smallest * (blues - greens) * _CustomReciprocal(math_ops.square(blues - reds)))
    dh_dr_5 = 60 * (math_ops.cast(blues > 0, dtype) * blue_biggest * green_smallest * _CustomReciprocal(blues - greens))
    dh_dr = dh_dr_1 + dh_dr_2 + dh_dr_3 + dh_dr_4 + dh_dr_5
    dh_dr = dh_dr / 360
    dh_dg_1 = 60 * (math_ops.cast(greens > 0, dtype) * green_biggest * -1 * (blues - reds) * _CustomReciprocal(math_ops.square(saturation)) * _CustomReciprocal(math_ops.square(value)))
    dh_dg_2 = 60 * (math_ops.cast(reds > 0, dtype) * red_biggest * green_smallest * (reds - blues) * _CustomReciprocal(math_ops.square(reds - greens)))
    dh_dg_3 = 60 * (math_ops.cast(reds > 0, dtype) * red_biggest * blue_smallest * _CustomReciprocal(reds - blues))
    dh_dg_4 = 60 * (math_ops.cast(blues > 0, dtype) * blue_biggest * green_smallest * (reds - blues) * _CustomReciprocal(math_ops.square(blues - greens)))
    dh_dg_5 = 60 * (math_ops.cast(blues > 0, dtype) * blue_biggest * red_smallest * -1 * _CustomReciprocal(blues - reds))
    dh_dg = dh_dg_1 + dh_dg_2 + dh_dg_3 + dh_dg_4 + dh_dg_5
    dh_dg = dh_dg / 360
    dh_db_1 = 60 * (math_ops.cast(blues > 0, dtype) * blue_biggest * -1 * (reds - greens) * _CustomReciprocal(math_ops.square(saturation)) * _CustomReciprocal(math_ops.square(value)))
    dh_db_2 = 60 * (math_ops.cast(reds > 0, dtype) * red_biggest * blue_smallest * (greens - reds) * _CustomReciprocal(math_ops.square(reds - blues)))
    dh_db_3 = 60 * (math_ops.cast(reds > 0, dtype) * red_biggest * green_smallest * -1 * _CustomReciprocal(reds - greens))
    dh_db_4 = 60 * (math_ops.cast(greens > 0, dtype) * green_biggest * blue_smallest * (greens - reds) * _CustomReciprocal(math_ops.square(greens - blues)))
    dh_db_5 = 60 * (math_ops.cast(greens > 0, dtype) * green_biggest * red_smallest * _CustomReciprocal(greens - reds))
    dh_db = dh_db_1 + dh_db_2 + dh_db_3 + dh_db_4 + dh_db_5
    dh_db = dh_db / 360
    dv_drgb = array_ops_stack.stack([grad[..., 2] * dv_dr, grad[..., 2] * dv_dg, grad[..., 2] * dv_db], axis=-1)
    ds_drgb = array_ops_stack.stack([grad[..., 1] * ds_dr, grad[..., 1] * ds_dg, grad[..., 1] * ds_db], axis=-1)
    dh_drgb = array_ops_stack.stack([grad[..., 0] * dh_dr, grad[..., 0] * dh_dg, grad[..., 0] * dh_db], axis=-1)
    gradient_input = math_ops.add(math_ops.add(dv_drgb, ds_drgb), dh_drgb)
    return gradient_input