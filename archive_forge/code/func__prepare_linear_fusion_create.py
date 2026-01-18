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
def _prepare_linear_fusion_create(cls, x: 'TensorBox', weight: 'TensorBox', bias: 'TensorBox'):
    """
    This function is a helper function to prepare inputs, layout and constant args
    for linear post-op fusion's create function. The function only supports the CPU device
    since linear post-op fusion kernel is only supported on CPU right now.
    """
    x.realize()
    weight.realize()
    if bias is not None:
        bias.realize()
    with V.graph.fake_mode:
        x_fake = ir_node_to_tensor(x, guard_shape=True)
        weight_fake = ir_node_to_tensor(weight, guard_shape=True)
        bias_fake = ir_node_to_tensor(bias, guard_shape=True) if bias is not None else bias
        if bias is not None:
            output = torch.ops.aten.addmm.default(bias_fake, x_fake, weight_fake)
        else:
            output = torch.ops.aten.mm.default(x_fake, weight_fake)
        output_size = output.size()
        req_stride_order = [1, 0]
        output_stride = make_contiguous_strides_for(output_size)
    x = cls.require_stride_order(x, req_stride_order)
    assert x.get_device().type == 'cpu' and weight.get_device().type == 'cpu'
    inputs = [x, weight]
    kernel_layout = FixedLayout(x.get_device(), x.get_dtype(), convert_shape_to_inductor(output_size), convert_shape_to_inductor(output_stride))
    constant_args: List[Any] = []
    if bias is not None:
        inputs.append(bias)
    else:
        constant_args.insert(0, bias)
    return (inputs, constant_args, kernel_layout, req_stride_order)