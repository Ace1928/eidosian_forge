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