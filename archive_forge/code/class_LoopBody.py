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
class LoopBody:
    """
    Captures the body of a Loops subclass into an FX graph.  Persists any
    indexing simplifications and makes it easier to analyze loop bodies.
    """

    def __init__(self, fn, args, var_ranges):
        super().__init__()
        self.var_ranges = var_ranges
        self.indexing_exprs = {}
        self.indexing_exprs_name = {}
        self.reads = []
        self.writes = []
        self.reads_name2expr = {}
        self.writes_name2expr = {}
        self.other = []
        self.submodules = {'get_index': self.get_index}
        self.subblocks = {}
        self.indirect_vars = []
        self.root_block = LoopBodyBlock(self, fn, args)
        self.indexing = None

    @cache_on_self
    def get_nodes(self):
        all_graphs = itertools.chain((self.root_block.graph,), (block.graph for block in self.subblocks.values()))
        return [node for graph in all_graphs for node in graph.nodes]

    @cache_on_self
    def bounds(self):
        from .bounds import BoundVars
        return BoundVars(self)

    def debug_str(self):
        lines = [f'var_ranges = {dict(self.var_ranges)}']
        lines.extend([f'{name} = {val}' for name, val in self.indexing_exprs.items()])
        lines.extend([block.debug_str(name) for name, block in itertools.chain([('body', self.root_block)], self.subblocks.items())])
        return '\n'.join(lines)

    def add_index_expr(self, expr: sympy.Expr, category, buf_name):
        getattr(self, category).append(expr)
        if buf_name is not None:
            getattr(self, f'{category}_name2expr')[buf_name] = expr
        if expr not in self.indexing_exprs_name:
            name = f'index{len(self.indexing_exprs)}'
            self.indexing_exprs_name[expr] = name
            self.indexing_exprs[name] = expr
        return self.indexing_exprs_name[expr]

    def add_submodule(self, block, prefix):
        """Not actually for nn.Modules, but subblocks in generated code are mapped to FX call_module opcodes"""
        if prefix[-1].isnumeric() and prefix not in self.submodules:
            name = prefix
        else:
            name = f'{prefix}{len(self.submodules)}'
        self.submodules[name] = block
        return name

    def add_indirect(self, size):
        name = f'indirect{len(self.indirect_vars)}'
        var = sympy_symbol(name)
        self.indirect_vars.append(var)
        return var

    def replace_indirect(self, old, new):
        """Swap in a variable used in indirect indexing"""
        if str(old) == str(new):
            return
        assert self.indexing is not None
        self.indexing = {k: sympy_subs(v, {old: new}) for k, v in self.indexing.items()}

    def get_index(self, name):
        assert self.indexing is not None
        return self.indexing[name]

    def __call__(self, *indices):
        index = list(itertools.chain(*indices))
        assert len(index) == len(self.var_ranges), (index, self.var_ranges)
        assert all((v not in self.var_ranges for v in index))
        replacements = dict(zip(self.var_ranges.keys(), index))
        self.indexing = {name: sympy_subs(expr, replacements) for name, expr in self.indexing_exprs.items()}
        result = self.root_block()
        self.indexing = None
        return result