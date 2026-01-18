import collections
import contextlib
import dataclasses
import functools
import itertools
import logging
import re
import textwrap
import traceback
from contextlib import nullcontext
from enum import Enum
from functools import partial
from inspect import signature
from typing import (
from unittest.mock import patch
import sympy
from sympy import Expr, Integer
import torch._export.serde.schema as export_schema
import torch._logging
import torch.fx
import torch.utils._pytree as pytree
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.utils import identity
from torch._export.serde.serialize import GraphModuleSerializer
from torch._prims_common import (
from torch._subclasses.fake_tensor import get_schema_info
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols, SymTypes
from torch.utils._sympy.functions import CleanDiv, FloorDiv, ModularIndexing
from . import config, dependencies
from .codegen.common import index_prevent_reordering
from .dependencies import (
from .utils import (
from .virtualized import ops, V
def _prepare_convolution_fusion_create(cls, x: 'TensorBox', weight: 'TensorBox', bias: 'TensorBox', padding: List[int], stride: List[int], dilation: List[int], groups: int, transposed: bool=False, output_padding: Optional[List[int]]=None):
    """
    This function is a helper function to prepare inputs, layout and constant args
    for convolution post-op fusion's create function, including deciding the output
    layout (channels first or channels last), realizing inputs and make them etc. The
    function only supports the CPU device since conv post-op fusion kernel is only
    supported on CPU right now.
    """

    def _conv_input_size(output_size, weight_size, padding, output_padding, stride, dilation, groups):
        assert len(output_size) == len(weight_size), 'Expect input dim == weight dim'
        dim = len(output_size)
        assert dim > 2, 'Expect input dim > 2'
        BATCH_DIM = 0
        WEIGHT_INPUT_CHANNELS_DIM = 1
        input_size = []
        input_size.append(output_size[BATCH_DIM])
        input_size.append(weight_size[WEIGHT_INPUT_CHANNELS_DIM] * groups)
        for d in range(2, dim):
            kernel = (weight_size[d] - 1) * dilation[d - 2] + 1
            input_size_d = (output_size[d] - 1) * stride[d - 2] - padding[d - 2] * 2 + kernel + output_padding[d - 2]
            input_size.append(input_size_d)
        return list(map(int, input_size))

    def _original_deconv_weight_size(prepacked_weight, groups):
        prepacked_weight_size = prepacked_weight.size()
        dim = len(prepacked_weight_size)
        assert dim > 2, 'Expect weight dim > 2'
        if groups > 1:
            weight_size = []
            weight_size.append(prepacked_weight_size[1] * groups)
            weight_size.append(prepacked_weight_size[0] / groups)
            for d in range(2, dim):
                weight_size.append(prepacked_weight_size[d])
        else:
            weight_size = prepacked_weight.transpose(0, 1).size()
        return weight_size
    x.realize()
    weight.realize()
    if bias is not None:
        bias.realize()
    with V.graph.fake_mode:
        x_fake = ir_node_to_tensor(x, guard_shape=True)
        weight_fake = ir_node_to_tensor(weight, guard_shape=True)
        dims = len(x_fake.size()) - 2
        assert 0 < len(padding) <= dims
        assert 0 < len(dilation) <= dims
        assert 0 < len(stride) <= dims
        padding = pad_listlike(padding, dims)
        dilation = pad_listlike(dilation, dims)
        stride = pad_listlike(stride, dims)
        if output_padding is None:
            output_padding = pad_listlike([0], dims)
        else:
            assert 0 < len(output_padding) <= dims
            output_padding = pad_listlike(output_padding, dims)
        assert isinstance(groups, int)
        if transposed:
            weight_size = _original_deconv_weight_size(weight_fake, groups)
            input_size = x_fake.size()
            output_size = _conv_input_size(input_size, weight_size, padding, output_padding, stride, dilation, groups)
        else:
            bias_fake = ir_node_to_tensor(bias, guard_shape=True) if bias is not None else bias
            output = torch.ops.aten.convolution(x_fake, weight_fake, bias_fake, stride, padding, dilation, transposed, output_padding, groups)
            output_size = output.size()
        req_stride_order = [0] + list(reversed(range(1, len(stride) + 1)))
        req_stride_order = [len(req_stride_order)] + req_stride_order
        output_stride = make_channels_last_strides_for(output_size)
    x = cls.require_stride_order(x, req_stride_order)
    assert x.get_device().type == 'cpu' and weight.get_device().type == 'cpu'
    inputs = [x, weight]
    kernel_layout = FixedLayout(x.get_device(), x.get_dtype(), convert_shape_to_inductor(output_size), convert_shape_to_inductor(output_stride))
    constant_args = [padding, stride, dilation, groups]
    if transposed:
        constant_args.insert(1, output_padding)
    if bias is not None:
        inputs.append(bias)
    else:
        constant_args.insert(0, bias)
    return (inputs, constant_args, kernel_layout, req_stride_order)