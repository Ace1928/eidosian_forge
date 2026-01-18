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
class MkldnnRnnLayer(ExternKernelAlloc):

    def __init__(self, layout, inputs, constant_args=()):
        super().__init__(layout, inputs, constant_args, None, kernel='aten.mkldnn_rnn_layer', cpp_kernel='at::mkldnn_rnn_layer')

    @classmethod
    def create(cls, x: 'TensorBox', w0: 'TensorBox', w1: 'TensorBox', w2: 'TensorBox', w3: 'TensorBox', hx: 'TensorBox', cx: 'TensorBox', reverse: bool, batch_sizes: List[int], mode: int, hidden_size: int, num_layers: int, has_biases: bool, bidirectional: bool, batch_first: bool, train: bool):
        x = cls.require_stride1(cls.realize_input(x))
        x.freeze_layout()
        w0 = cls.require_stride1(cls.realize_input(w0))
        w1 = cls.require_stride1(cls.realize_input(w1))
        w2 = cls.require_stride1(cls.realize_input(w2))
        w3 = cls.require_stride1(cls.realize_input(w3))
        hx = cls.require_stride1(cls.realize_input(hx))
        hx.freeze_layout()
        cx = cls.require_stride1(cls.realize_input(cx))
        cx.freeze_layout()
        input_size = x.get_size()
        assert len(input_size) == 3, 'Expect lstm input to be 3D'
        seq_length, mini_batch, input_size = input_size
        output_shape = [seq_length, mini_batch, hidden_size]
        hy_shape = hx.get_size()
        cy_shape = cx.get_size()
        res: List[IRNode] = []
        inputs = [x, w0, w1, w2, w3, hx, cx]
        constant_args = [reverse, batch_sizes, mode, hidden_size, num_layers, has_biases, bidirectional, batch_first, train]
        packed = MkldnnRnnLayer(MultiOutputLayout(x.get_device()), inputs=inputs, constant_args=constant_args)

        def get_strides_of_lstm_output(output_shape, batch_first):
            assert len(output_shape) == 3, 'Expect output_shape to be 3D'
            return make_contiguous_strides_for(output_shape)
        output_sizes = [output_shape, hy_shape, cy_shape]
        output_strides = [get_strides_of_lstm_output(output_shape, batch_first), make_contiguous_strides_for(hy_shape), make_contiguous_strides_for(cy_shape)]
        output_ir = [MultiOutput(FixedLayout(x.get_device(), x.get_dtype(), output_size, output_stride), packed, [(tuple, i)]) for i, (output_size, output_stride) in enumerate(zip(output_sizes, output_strides))]
        return output_ir