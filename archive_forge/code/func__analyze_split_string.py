import re
import tensorflow.compat.v2 as tf
from keras.src import activations
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.engine.base_layer import Layer
from tensorflow.python.util.tf_export import keras_export
def _analyze_split_string(split_string, bias_axes, input_shape, output_shape, left_elided=False):
    """Analyze an pre-split einsum string to find the weight shape."""
    input_spec = split_string.group(1)
    weight_spec = split_string.group(2)
    output_spec = split_string.group(3)
    elided = len(input_shape) - len(input_spec)
    if isinstance(output_shape, int):
        output_shape = [output_shape]
    else:
        output_shape = list(output_shape)
    output_shape.insert(0, input_shape[0])
    if elided > 0 and left_elided:
        for i in range(1, elided):
            output_shape.insert(1, input_shape[i])
    elif elided > 0 and (not left_elided):
        for i in range(len(input_shape) - elided, len(input_shape)):
            output_shape.append(input_shape[i])
    if left_elided:
        input_dim_map = {dim: i + elided - len(input_shape) for i, dim in enumerate(input_spec)}
        output_dim_map = {dim: i + elided for i, dim in enumerate(output_spec)}
    else:
        input_dim_map = {dim: i for i, dim in enumerate(input_spec)}
        output_dim_map = {dim: i for i, dim in enumerate(output_spec)}
    for dim in input_spec:
        input_shape_at_dim = input_shape[input_dim_map[dim]]
        if dim in output_dim_map:
            output_shape_at_dim = output_shape[output_dim_map[dim]]
            if output_shape_at_dim is not None and output_shape_at_dim != input_shape_at_dim:
                raise ValueError(f"Input shape and output shape do not match at shared dimension '{dim}'. Input shape is {input_shape_at_dim}, and output shape is {output_shape[output_dim_map[dim]]}.")
    for dim in output_spec:
        if dim not in input_spec and dim not in weight_spec:
            raise ValueError(f"Dimension '{dim}' was specified in the output '{output_spec}' but has no corresponding dim in the input spec '{input_spec}' or weight spec '{output_spec}'")
    weight_shape = []
    for dim in weight_spec:
        if dim in input_dim_map:
            weight_shape.append(input_shape[input_dim_map[dim]])
        elif dim in output_dim_map:
            weight_shape.append(output_shape[output_dim_map[dim]])
        else:
            raise ValueError(f"Weight dimension '{dim}' did not have a match in either the input spec '{input_spec}' or the output spec '{output_spec}'. For this layer, the weight must be fully specified.")
    if bias_axes is not None:
        num_left_elided = elided if left_elided else 0
        idx_map = {char: output_shape[i + num_left_elided] for i, char in enumerate(output_spec)}
        for char in bias_axes:
            if char not in output_spec:
                raise ValueError(f"Bias dimension '{char}' was requested, but is not part of the output spec '{output_spec}'")
        first_bias_location = min([output_spec.find(char) for char in bias_axes])
        bias_output_spec = output_spec[first_bias_location:]
        bias_shape = [idx_map[char] if char in bias_axes else 1 for char in bias_output_spec]
        if not left_elided:
            for _ in range(elided):
                bias_shape.append(1)
    else:
        bias_shape = None
    return (weight_shape, bias_shape, output_shape)