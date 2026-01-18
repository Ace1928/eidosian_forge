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
class AllReduceCoalesced(InPlaceCollectiveKernel):

    def __init__(self, layout, inputs, constant_args, reduce_op):
        super().__init__(layout, inputs, constant_args)
        self.reduce_op = reduce_op

    def should_allocate(self):
        return False

    def get_mutation_names(self):
        return [self.inputs[0].get_name()]

    def get_unbacked_symbol_defs(self):
        return {}

    @classmethod
    def create(cls, inputs: List['TensorBox'], reduce_op: str, tag: str, ranks: List[int], group_size: int):
        inplace_inputs = cls.wrap_inputs_as_inplace(inputs)
        packed = AllReduceCoalesced(layout=NoneLayout(inplace_inputs[0].get_device()), inputs=inplace_inputs, constant_args=[tag, ranks, group_size], reduce_op=reduce_op)
        mark_node_as_mutating(packed, inplace_inputs[0])
        return inplace_inputs

    def codegen_collective(self, wrapper, output_name, input_names):
        wrapper.writeline(f"{output_name}_work = dist.all_reduce_coalesced({output_name}, op=fun_col_impl._str_to_reduce_op('{str(self.reduce_op)}'), group={output_name}_pg, async_op=True)")