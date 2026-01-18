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
@symbolic_helper.parse_args('v', 'v', 'v', 'v', 'i', 'i', 'f', 'i', 'i')
def _rnn_packed(g, input, batch_sizes, hidden, weight_v, has_biases, num_layers, dropout, train, bidirectional):
    weight = symbolic_helper._unpack_list(weight_v)
    return _generic_rnn(g, kind, input, hidden, weight, has_biases, num_layers, dropout, train, bidirectional, batch_sizes=batch_sizes)