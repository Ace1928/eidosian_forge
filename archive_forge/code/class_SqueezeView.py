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
class SqueezeView(BaseView):

    @classmethod
    def create(cls, x, *, dim=None):
        if is_storage_and_layout(x):
            storage, old_layout = as_storage_and_layout(x)
            new_size = []
            new_stride = []
            if dim is not None:
                assert isinstance(dim, int), 'expected integer dim argument'
                assert 0 <= dim and dim < len(old_layout.size)
            for i, (size, stride) in enumerate(zip(old_layout.size, old_layout.stride)):
                if dim is None:
                    if size != 1:
                        new_size.append(size)
                        new_stride.append(stride)
                elif i != dim:
                    new_size.append(size)
                    new_stride.append(stride)
                else:
                    assert size == 1, 'expected squeezed size to be 1'
            new_layout = FixedLayout(old_layout.device, old_layout.dtype, new_size, new_stride, old_layout.offset)
            return ReinterpretView(storage, new_layout)
        if dim is None:
            return View.create(x, [s for s in x.get_size() if s != 1])
        else:
            assert x.get_size()[dim] == 1
            return View.create(x, [s for i, s in enumerate(x.get_size()) if i != dim])

    @staticmethod
    def squeezer(size: Tuple[sympy.Expr, ...]):
        new_size = [s for s in size if s != 1]
        not_one = [i for i, s in enumerate(size) if s != 1]
        length = len(size)

        def reindex(index: List[sympy.Expr]) -> Tuple[sympy.Expr, ...]:
            assert len(index) == len(not_one), f'{index} {not_one}'
            new_index = [sympy.Integer(0)] * length
            for idx, s in zip(not_one, index):
                new_index[idx] = s
            return tuple(new_index)
        return (new_size, reindex)

    def __init__(self, data):
        raise AssertionError('use SqueezeView.create()')