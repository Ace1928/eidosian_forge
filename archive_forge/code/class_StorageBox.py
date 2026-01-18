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
class StorageBox(MutableBox):

    def is_input_buffer(self):
        if isinstance(self.data, (InputBuffer, ReinterpretView)):
            return self.data.get_name() in V.graph.graph_inputs
        return False

    def realize(self):
        if isinstance(self.data, (ComputedBuffer, InputsKernel, InputBuffer, ReinterpretView, TemplateBuffer)):
            return self.data.get_name()
        assert isinstance(self.data, (Pointwise, Reduction)), type(self.data)
        origin_node = self.data.get_origin_node()
        traceback = self.data.get_traceback()
        self.data = ComputedBuffer(name=None, layout=FlexibleLayout(device=self.data.get_device(), dtype=self.data.get_dtype(), size=self.data.get_size()), data=self.data)
        self.data.name = V.graph.register_buffer(self.data)
        self.data.origins = self.origins
        self.data.origin_node = origin_node
        self.data.traceback = traceback
        return self.data.name

    def realize_hint(self):
        """
        Called on buffers we expect to be forced to realize later.
        """
        if isinstance(self.data, (Pointwise, Reduction)) and self.num_reads() > 1 and self.is_pointwise_non_scalar_tensor_num_reads_larger_than_one():
            self.realize()

    def has_exceeded_max_reads(self):
        return isinstance(self.data, Pointwise) and (self.num_reads() > config.realize_acc_reads_threshold or self.inner_fn_str_len() > config.realize_bytes_threshold)

    def mark_reuse(self, users):
        """
        A heuristic to decide if we should realize a tensor
        that is used multiple times.
        """

        def should_realize_on_cpu(loops: Union[Pointwise, Reduction]):
            """
            The heuristic for realizing reused result of heavy ops on cpu
            """
            heavy_ops = ['exp']
            fn_str = loops.inner_fn_str()
            return any((op + '(' in fn_str for op in heavy_ops))
        if users > 1 and isinstance(self.data, (Pointwise, Reduction)) and (self.num_reads() > config.realize_reads_threshold or len(self.inner_fn_str()) > config.realize_bytes_threshold or (is_cpu(self.data) and should_realize_on_cpu(self.data))):
            self.realize()

    @cache_on_self
    def num_reads(self):
        data = self.data
        if isinstance(data, (InputsKernel, InputBuffer, ReinterpretView)):
            return 1
        if isinstance(data, ComputedBuffer):
            read_writes = data.get_read_writes()
        else:
            assert isinstance(data, (Pointwise, Reduction)), type(data)
            read_writes = ComputedBuffer(name=None, layout=FlexibleLayout(device=data.get_device(), dtype=data.get_dtype(), size=data.get_size()), data=data).get_read_writes()
        return len(read_writes.reads)

    @cache_on_self
    def is_pointwise_non_scalar_tensor_num_reads_larger_than_one(self):
        return sum((read.index != 0 for read in self.data.get_reads())) > 1 if isinstance(self.data, Pointwise) and all((not isinstance(read, dependencies.StarDep) for read in self.data.get_reads())) else True