import dataclasses
import itertools
import operator
from typing import Any, Callable, Dict, List, Tuple
import torch
from torch.fx import Graph, GraphModule, Node
from torch.fx.subgraph_rewriter import (
import torch.nn.functional as F
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch.ao.quantization.quantizer import (
from .utils import (
def _quantized_qat_conv_bn_pattern(x: torch.Tensor, conv_weight: torch.Tensor, bn_weight: torch.Tensor, bn_bias: torch.Tensor, bn_running_mean: torch.Tensor, bn_running_var: torch.Tensor, **kwargs) -> torch.Tensor:
    running_std = torch.sqrt(bn_running_var + bn_eps)
    scale_factor = bn_weight / running_std
    weight_shape = [1] * len(conv_weight.shape)
    weight_shape[0] = -1
    bias_shape = [1] * len(conv_weight.shape)
    bias_shape[1] = -1
    scaled_weight = conv_weight * scale_factor.reshape(weight_shape)
    scaled_weight = _append_qdq(scaled_weight, is_per_channel, kwargs)
    if has_bias:
        zero_bias = torch.zeros_like(kwargs['conv_bias'], dtype=x.dtype)
        if bias_is_quantized:
            zero_bias = _append_qdq(zero_bias, is_per_channel, kwargs)
        x = conv_fn(x, scaled_weight, zero_bias)
    else:
        x = conv_fn(x, scaled_weight, None)
    x = x / scale_factor.reshape(bias_shape)
    if has_bias:
        x = x + kwargs['conv_bias'].reshape(bias_shape)
    x = F.batch_norm(x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=bn_is_training, eps=bn_eps)
    return x