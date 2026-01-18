import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
def _group_normalization(x, num_groups, scale, bias, epsilon=1e-05):
    assert x.shape[1] % num_groups == 0
    group_size = x.shape[1] // num_groups
    new_shape = [x.shape[0], num_groups, group_size, *list(x.shape[2:])]
    x_reshaped = x.reshape(new_shape)
    axes = tuple(range(2, len(new_shape)))
    mean = np.mean(x_reshaped, axis=axes, keepdims=True)
    var = np.var(x_reshaped, axis=axes, keepdims=True)
    x_normalized = ((x_reshaped - mean) / np.sqrt(var + epsilon)).reshape(x.shape)
    dim_ones = (1,) * (len(x.shape) - 2)
    scale = scale.reshape(-1, *dim_ones)
    bias = bias.reshape(-1, *dim_ones)
    return scale * x_normalized + bias