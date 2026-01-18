import re
import string
import numpy as np
from keras.src import activations
from keras.src import backend
from keras.src import constraints
from keras.src import dtype_policies
from keras.src import initializers
from keras.src import ops
from keras.src import quantizers
from keras.src import regularizers
from keras.src.api_export import keras_export
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
def _analyze_quantization_info(equation, input_shape):

    def get_specs(equation, input_shape):
        possible_labels = string.ascii_letters
        dot_replaced_string = re.sub('\\.\\.\\.', '0', equation)
        split_string = re.match('([a-zA-Z]+),([a-zA-Z]+)->([a-zA-Z]+)', dot_replaced_string)
        if split_string is not None:
            input_spec = split_string.group(1)
            weight_spec = split_string.group(2)
            output_spec = split_string.group(3)
            return (input_spec, weight_spec, output_spec)
        split_string = re.match('0([a-zA-Z]+),([a-zA-Z]+)->0([a-zA-Z]+)', dot_replaced_string)
        if split_string is not None:
            input_spec = split_string.group(1)
            weight_spec = split_string.group(2)
            output_spec = split_string.group(3)
            elided = len(input_shape) - len(input_spec)
            possible_labels = sorted(set(possible_labels) - set(input_spec) - set(weight_spec) - set(output_spec))
            for i in range(elided):
                input_spec = possible_labels[i] + input_spec
                output_spec = possible_labels[i] + output_spec
            return (input_spec, weight_spec, output_spec)
        split_string = re.match('([a-zA-Z]{2,})0,([a-zA-Z]+)->([a-zA-Z]+)0', dot_replaced_string)
        if split_string is not None:
            input_spec = split_string.group(1)
            weight_spec = split_string.group(2)
            output_spec = split_string.group(3)
            elided = len(input_shape) - len(input_spec)
            possible_labels = sorted(set(possible_labels) - set(input_spec) - set(weight_spec) - set(output_spec))
            for i in range(elided):
                input_spec = input_spec + possible_labels[i]
                output_spec = output_spec + possible_labels[i]
            return (input_spec, weight_spec, output_spec)
        raise ValueError(f"Invalid einsum equation '{equation}'. Equations must be in the form [X],[Y]->[Z], ...[X],[Y]->...[Z], or [X]...,[Y]->[Z]....")
    input_spec, weight_spec, output_spec = get_specs(equation, input_shape)
    input_reduced_axes = []
    weight_reduced_axes = []
    for i, label in enumerate(input_spec):
        index = output_spec.find(label)
        if index == -1:
            input_reduced_axes.append(i)
    for i, label in enumerate(weight_spec):
        index = output_spec.find(label)
        if index == -1:
            weight_reduced_axes.append(i)
    input_expand_axes = []
    weight_expand_axes = []
    for i, label in enumerate(output_spec):
        index_input = input_spec.find(label)
        index_weight = weight_spec.find(label)
        if index_input == -1:
            input_expand_axes.append(i)
        if index_weight == -1:
            weight_expand_axes.append(i)
    input_transpose_axes = []
    weight_transpose_axes = []
    for i, label in enumerate(output_spec):
        index_input = input_spec.find(label)
        index_weight = weight_spec.find(label)
        if index_input != -1:
            input_transpose_axes.append(index_input)
        if index_weight != -1:
            weight_transpose_axes.append(index_weight)
    input_squeeze_axes = []
    weight_squeeze_axes = []
    for ori_index in input_reduced_axes:
        try:
            index = input_expand_axes.pop(0)
        except IndexError:
            input_squeeze_axes.append(ori_index)
        input_transpose_axes.insert(index, ori_index)
    for ori_index in weight_reduced_axes:
        try:
            index = weight_expand_axes.pop(0)
        except IndexError:
            weight_squeeze_axes.append(ori_index)
        weight_transpose_axes.insert(index, ori_index)
    custom_gradient_equation = f'{output_spec},{weight_spec}->{input_spec}'
    weight_reverse_transpose_axes = [i for _, i in sorted(((v, i) for i, v in enumerate(weight_transpose_axes)))]
    return (input_reduced_axes, weight_reduced_axes, input_transpose_axes, weight_transpose_axes, input_expand_axes, weight_expand_axes, input_squeeze_axes, weight_squeeze_axes, custom_gradient_equation, weight_reverse_transpose_axes)