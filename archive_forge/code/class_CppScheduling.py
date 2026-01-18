import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import re
import sys
from copy import copy, deepcopy
from typing import Dict, List, Optional, Set, Tuple, Union
import sympy
import torch
import torch.fx
from torch._inductor import dependencies
from torch._inductor.ir import StorageBox, TensorBox
from torch._prims_common import is_float_dtype
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges
from .. import codecache, config, ir, metrics
from ..codegen.wrapper import WrapperCodeGen
from ..optimize_indexing import range_expressable_in_32_bits
from ..scheduler import BaseScheduling, SchedulerNode
from ..utils import (
from ..virtualized import ops, V
from .common import (
class CppScheduling(BaseScheduling):
    MAX_FUSED_KERNEL_ARGS_NUM = 500

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.get_kernel_group()
        self._ready_to_flush = False

    def _set_flush_status(self, status: bool):
        self._ready_to_flush = status

    def group_fn(self, sizes):
        return tuple((tuple(map(V.graph.sizevars.simplify, s)) for s in sizes))

    def get_kernel_group(self):
        from .wrapper import CppWrapperCodeGen
        self.kernel_group: Union[CppWrapperKernelGroup, KernelGroup]
        if isinstance(V.graph.wrapper_code, CppWrapperCodeGen):
            self.kernel_group = CppWrapperKernelGroup()
        else:
            self.kernel_group = KernelGroup()

    def _can_fuse_horizontal_impl(self, node1, node2):
        _, (vars1, reduce1) = node1.group
        _, (vars2, reduce2) = node2.group
        if vars1 == vars2 and reduce1 == reduce2:
            return True
        if reduce1 == () and vars1 == vars2 + reduce2:
            return True
        return False

    def can_fuse_horizontal(self, node1, node2):
        if len(node1.get_nodes()) + len(node2.get_nodes()) > config.cpp.max_horizontal_fusion_size:
            return False
        return self._can_fuse_horizontal_impl(node1, node2)

    def can_fuse_vertical(self, node1, node2):
        return self._can_fuse_horizontal_impl(node1, node2) and (not node1.is_reduction())

    def codegen_nodes(self, nodes):
        """
        Turn an set of pre-fused nodes into a C++ kernel.
        """
        kernel_group = self.kernel_group
        cpp_kernel_proxy = CppKernelProxy(kernel_group)
        cpp_kernel_proxy.codegen_nodes(nodes)
        kernel_group.finalize_kernel(cpp_kernel_proxy, nodes)
        args_num = self._get_scheduled_num_args()
        if args_num > CppScheduling.MAX_FUSED_KERNEL_ARGS_NUM:
            self._set_flush_status(True)

    def _get_scheduled_num_args(self):
        return self.kernel_group.get_num_args()

    def ready_to_flush(self):
        return self._ready_to_flush

    def codegen_sync(self):
        pass

    def flush(self):
        self.kernel_group.codegen_define_and_call(V.graph.wrapper_code)
        self.get_kernel_group()
        self._set_flush_status(False)