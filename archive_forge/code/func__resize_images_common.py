import functools
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_case
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _resize_images_common(images, resizer_fn, size, preserve_aspect_ratio, name, skip_resize_if_same):
    """Core functionality for v1 and v2 resize functions."""
    with ops.name_scope(name, 'resize', [images, size]):
        images = ops.convert_to_tensor(images, name='images')
        if images.get_shape().ndims is None:
            raise ValueError("'images' contains no shape.")
        is_batch = True
        if images.get_shape().ndims == 3:
            is_batch = False
            images = array_ops.expand_dims(images, 0)
        elif images.get_shape().ndims != 4:
            raise ValueError("'images' must have either 3 or 4 dimensions.")
        _, height, width, _ = images.get_shape().as_list()
        try:
            size = ops.convert_to_tensor(size, dtypes.int32, name='size')
        except (TypeError, ValueError):
            raise ValueError("'size' must be a 1-D int32 Tensor")
        if not size.get_shape().is_compatible_with([2]):
            raise ValueError("'size' must be a 1-D Tensor of 2 elements: new_height, new_width")
        if preserve_aspect_ratio:
            _, current_height, current_width, _ = _ImageDimensions(images, rank=4)
            scale_factor_height = math_ops.cast(size[0], dtypes.float32) / math_ops.cast(current_height, dtypes.float32)
            scale_factor_width = math_ops.cast(size[1], dtypes.float32) / math_ops.cast(current_width, dtypes.float32)
            scale_factor = math_ops.minimum(scale_factor_height, scale_factor_width)
            scaled_height_const = math_ops.cast(math_ops.round(scale_factor * math_ops.cast(current_height, dtypes.float32)), dtypes.int32)
            scaled_width_const = math_ops.cast(math_ops.round(scale_factor * math_ops.cast(current_width, dtypes.float32)), dtypes.int32)
            size = ops.convert_to_tensor([scaled_height_const, scaled_width_const], dtypes.int32, name='size')
        size_const_as_shape = tensor_util.constant_value_as_shape(size)
        new_height_const = tensor_shape.dimension_at_index(size_const_as_shape, 0).value
        new_width_const = tensor_shape.dimension_at_index(size_const_as_shape, 1).value
        if skip_resize_if_same and all((x is not None for x in [new_width_const, width, new_height_const, height])) and (width == new_width_const and height == new_height_const):
            if not is_batch:
                images = array_ops.squeeze(images, axis=[0])
            return images
        images = resizer_fn(images, size)
        images.set_shape([None, new_height_const, new_width_const, None])
        if not is_batch:
            images = array_ops.squeeze(images, axis=[0])
        return images