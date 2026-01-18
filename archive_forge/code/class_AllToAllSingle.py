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
class AllToAllSingle(OutOfPlaceCollectiveKernel):

    def __init__(self, layout, inputs, outputs, constant_args, output_split_sizes, input_split_sizes):
        super().__init__(layout, inputs, outputs, constant_args)
        self.output_split_sizes = output_split_sizes
        self.input_split_sizes = input_split_sizes

    def get_unbacked_symbol_uses(self):
        r = set()
        if self.output_split_sizes is not None:
            r |= free_unbacked_symbols(self.output_split_sizes)
        if self.input_split_sizes is not None:
            r |= free_unbacked_symbols(self.input_split_sizes)
        return r

    @classmethod
    def create(cls, x: 'TensorBox', output_split_sizes: Optional[List[Expr]], input_split_sizes: Optional[List[Expr]], tag: str, ranks: List[int], group_size: int):
        inputs = [cls.realize_input(x)]

        def compute_size(new_size):
            if output_split_sizes is not None:
                new_size[0] = sum(output_split_sizes)
        outputs = cls.create_output_buffers(inputs, compute_size)
        layout = MultiOutputLayout(inputs[0].get_device())
        packed = AllToAllSingle(layout=layout, inputs=inputs, outputs=outputs, constant_args=[tag, ranks, group_size], output_split_sizes=output_split_sizes, input_split_sizes=input_split_sizes)
        return cls.create_output_nodes(packed, outputs)[0]

    def codegen_collective(self, wrapper, output_name, input_names):
        tag, ranks, group_size = self.constant_args
        wrapper.writeline(f'{output_name}_work = dist.all_to_all_single({output_name}[0], {output_name}_inputs[0], output_split_sizes={self.output_split_sizes}, input_split_sizes={self.input_split_sizes}, group={output_name}_pg, async_op=True)')