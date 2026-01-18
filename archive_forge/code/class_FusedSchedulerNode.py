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
class FusedSchedulerNode(BaseSchedulerNode):
    """
    This is a "fake" scheduler node that represents a group of scheduler nodes
    that are meant to be fused together. The way it does this is by maintaining
    its unmet dependencies as the union of its constituent nodes.
    """

    @classmethod
    def fuse(cls, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
        assert node1.scheduler is node2.scheduler
        assert isinstance(node1, (SchedulerNode, FusedSchedulerNode)) and isinstance(node2, (SchedulerNode, FusedSchedulerNode))
        return cls(node1.scheduler, list(node1.get_nodes()) + list(node2.get_nodes()))

    def __init__(self, scheduler: 'Scheduler', snodes: List[SchedulerNode]):
        self.snodes = snodes
        self.scheduler = scheduler
        self.node: ir.Buffer = None
        self.users: List[NodeUser] = []
        self.inverse_users = []
        self.node_users = []
        self.group = max(snodes, key=lambda x: int(x.is_reduction())).group
        self.ancestors = set.union(*[x.ancestors for x in snodes if x.ancestors is not None])
        self.set_read_writes(dependencies.ReadWrites.merge_list([x.read_writes for x in snodes]))
        self.unmet_dependencies = {dep for dep in set.union(*[x.unmet_dependencies for x in snodes]) if dep.name not in self.get_names()} - self.read_writes.writes
        self.min_order = min([x.min_order for x in self.snodes])
        self.max_order = max([x.max_order for x in self.snodes])

    @cache_on_self
    def get_name(self) -> str:
        return '_'.join([x.get_name() for x in self.snodes])

    def get_first_name(self) -> str:
        return self.snodes[0].get_name()

    @cache_on_self
    def get_names(self) -> Set[str]:
        return set.union(*[x.get_names() for x in self.snodes])

    def debug_str_extra(self) -> str:
        lines = [f'{self.get_name()}.snodes[{i}] =\n{node.debug_str()}' for i, node in enumerate(self.snodes)]
        return textwrap.indent('\n'.join(lines).rstrip(), '    ')

    def set_last_usage(self, future_used_buffers: Set[str], mutation_real_name: Dict[str, str]):
        super().set_last_usage(future_used_buffers, mutation_real_name)
        future_used_buffers: Set[str] = set()
        for node in reversed(self.snodes):
            node.set_last_usage(future_used_buffers, mutation_real_name)
            future_used_buffers.update(node.last_usage)

    @cache_on_self
    def used_buffer_names(self) -> Set[str]:
        return set.union(*[x.used_buffer_names() for x in self.snodes])

    @cache_on_self
    def used_or_aliased_buffer_names(self) -> Set[str]:
        return set.union(*[x.used_or_aliased_buffer_names() for x in self.snodes])

    def get_nodes(self) -> List[SchedulerNode]:
        return self.snodes

    def __repr__(self):
        return f'{type(self).__name__}(nodes={self.get_name()})'

    @cache_on_self
    def is_reduction(self):
        return any((x.is_reduction() for x in self.snodes))

    @cache_on_self
    def is_template(self):
        return any((x.is_template() for x in self.snodes))

    @cache_on_self
    def get_template_node(self):
        for node in self.snodes:
            if node.is_template():
                return node
        return None

    def get_device(self):
        return self.group[0]

    @cache_on_self
    def has_aliasing_or_mutation(self):
        return any((x.has_aliasing_or_mutation() for x in self.snodes))

    @cache_on_self
    def op_counts(self):
        op_counts: Counter[str] = collections.Counter()
        for node in self.snodes:
            op_counts.update(node.op_counts())
        return op_counts

    def has_atomic_add(self, check_buf):
        return any((isinstance(sub_schedule_node1, SchedulerNode) and sub_schedule_node1.has_atomic_add(check_buf) for sub_schedule_node1 in self.get_nodes()))

    def update_mutated_names(self, renames: Dict[str, str]):
        raise NotImplementedError

    def add_mutation_dep(self, name):
        raise NotImplementedError

    def set_users(self, users: List['NodeUser']):
        raise NotImplementedError

    def get_aliases(self):
        raise NotImplementedError

    def get_mutations(self):
        raise NotImplementedError

    def can_inplace(self, read_dep: dependencies.MemoryDep):
        raise NotImplementedError

    def allocate(self):
        raise NotImplementedError

    def can_free(self):
        raise NotImplementedError