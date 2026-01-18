import collections
import dataclasses
import functools
import itertools
import logging
import math
import os
import pprint
import textwrap
from typing import (
import sympy
import torch
from torch._dynamo.utils import dynamo_timed
from torch._inductor.metrics import get_metric_table, is_metric_table_enabled
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.utils._triton import has_triton
from . import comms, config, dependencies, ir, metrics
from .codegen.common import get_scheduling_for_device, Kernel
from .comm_analysis import estimate_nccl_collective_runtime
from .dependencies import StarDep, WeakDep
from .ir import ComputedBuffer, MultiOutput, MultiOutputLayout
from .sizevars import SimplifyIndexing
from .utils import (
from .virtualized import V
class BaseScheduling:

    def can_fuse_vertical(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
        """
        Check whether node1 and node2 can be vertically fused or not.
        """
        raise NotImplementedError()

    def can_fuse_horizontal(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
        """
        Check whether node1 and node2 can be horizontally fused or not.
        """
        raise NotImplementedError()

    def group_fn(self, sizes):
        """
        Process the iteration sizes in case a transformation needs to be applied.
        """
        raise NotImplementedError()

    def codegen_template(self, template_node: SchedulerNode, epilogue_nodes: List[SchedulerNode]):
        """
        Given a template node, generate a kernel.

        This function is only available for triton now. If the third-party backend behaves as a sub-class
        of TritonScheduling, it can override it or reuse it.
        """
        raise NotImplementedError()

    def codegen_nodes(self, nodes: List[BaseSchedulerNode]):
        """
        Generate a kernel given a list of pre-fused nodes.
        """
        raise NotImplementedError()

    def codegen_sync(self):
        """
        Generate synchronization code for the kernel. This method depends on the hardware characteristics.
        """
        raise NotImplementedError()

    def ready_to_flush(self) -> bool:
        """
        Check whether the backend is requesting the scheduler to flush the generated kernel.
        If not supported, please return False.
        """
        return False

    def flush(self):
        """
        Flush the generated kernel and python wrapper code to the source code file.
        """
        raise NotImplementedError()

    def benchmark_fused_nodes(self, nodes):
        """
        Benchmark fused list of nodes and return the execution time
        in milliseconds on randomly generated inputs.
        """
        raise NotImplementedError()