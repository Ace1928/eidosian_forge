from __future__ import annotations
import builtins
import functools
import math
import sys
import warnings
from typing import Callable, List, Optional, Sequence, Tuple, Union
import torch
import torch._C._onnx as _C_onnx
import torch.nn.modules.utils
import torch.onnx
from torch import _C
from torch.onnx import _constants, _deprecation, _type_utils, errors, symbolic_helper
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils, registration
from torch.types import Number
@_onnx_symbolic('aten::adaptive_avg_pool1d', decorate=[_apply_params('adaptive_avg_pool1d', 'AveragePool', torch.nn.modules.utils._single), _export('adaptive_avg_pool1d')])
@_onnx_symbolic('aten::adaptive_avg_pool2d', decorate=[_apply_params('adaptive_avg_pool2d', 'AveragePool', torch.nn.modules.utils._pair), _export('adaptive_avg_pool2d')])
@_onnx_symbolic('aten::adaptive_avg_pool3d', decorate=[_apply_params('adaptive_avg_pool3d', 'AveragePool', torch.nn.modules.utils._triple), _export('adaptive_avg_pool3d')])
@_onnx_symbolic('aten::adaptive_max_pool1d', decorate=[_apply_params('adaptive_max_pool1d', 'MaxPool', torch.nn.modules.utils._single, max_pool1d_with_indices), _export('adaptive_max_pool1d')])
@_onnx_symbolic('aten::adaptive_max_pool2d', decorate=[_apply_params('adaptive_max_pool2d', 'MaxPool', torch.nn.modules.utils._pair, max_pool2d_with_indices), _export('adaptive_max_pool2d')])
@_onnx_symbolic('aten::adaptive_max_pool3d', decorate=[_apply_params('adaptive_max_pool3d', 'MaxPool', torch.nn.modules.utils._triple, max_pool3d_with_indices), _export('adaptive_max_pool3d')])
@_beartype.beartype
def _adaptive_pool(name, type, tuple_fn, fn=None):

    @symbolic_helper.quantized_args(True, False)
    @_beartype.beartype
    def symbolic_fn(g, input, output_size):
        output_size_value = output_size
        try:
            output_size = symbolic_helper._parse_arg(output_size, 'is')
        except Exception:
            return symbolic_helper._onnx_unsupported('adaptive pooling, since output_size is not constant.', input)
        if output_size == [1] * len(output_size) and type == 'AveragePool':
            return g.op('GlobalAveragePool', input)
        sizes = symbolic_helper._get_tensor_sizes(input)
        try:
            dim = sizes[2:]
        except Exception:
            dim = None
        if dim is None or any((i is None for i in dim)):
            if output_size == [1] * len(output_size):
                return (g.op('GlobalMaxPool', input), None)
            return symbolic_helper._unimplemented(name, 'input size not accessible', input)
        mod = [dim[i] % output_size[i] for i in range(0, len(dim))]
        if mod != [0] * len(mod):
            if output_size == [1] * len(output_size):
                return (g.op('GlobalMaxPool', input), None)
            return symbolic_helper._unimplemented(name, 'output size that are not factor of input size', output_size_value)
        k = [int(dim[i] / output_size[i]) for i in range(0, len(dim))]
        if type == 'MaxPool':
            return fn(g, input, k, k, (0,) * len(dim), (1,) * len(dim), False)
        output = g.op(type, input, kernel_shape_i=tuple_fn(k), strides_i=tuple_fn(k))
        return output
    return symbolic_fn