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
class SchedulerNode(BaseSchedulerNode):

    def __init__(self, scheduler: 'Scheduler', node: Union[ir.ComputedBuffer, ir.TemplateBuffer], group_fn):
        super().__init__(scheduler, node)
        self._sizes, self._body = node.simplify_and_reorder()
        self.group = (node.get_device(), group_fn(self._sizes))
        if isinstance(node, ir.TemplateBuffer):
            self.set_read_writes(node.normalized_read_writes())
        else:
            self.set_read_writes(dependencies.extract_read_writes(self._body, *self._sizes, normalize=True))

    def debug_str_extra(self) -> str:
        name = self.get_name()
        lines = [f'{name}.group.device = {self.group[0]}', f'{name}.group.iteration = {self.group[1]}', f'{name}.sizes = {self._sizes}']
        if self.get_aliases():
            lines.append(f'{name}.aliases = {pformat(self.get_aliases())}')
        if self.get_mutations():
            lines.append(f'{name}.mutations = {pformat(self.get_mutations())}')
        if isinstance(self._body, ir.LoopBody):
            lines.append(f'class {name}_loop_body:')
            lines.append(textwrap.indent(self._body.debug_str(), '    '))
        return '\n'.join(lines)

    def get_ranges(self):
        return self._sizes

    def is_reduction(self):
        assert isinstance(self.node, (ir.ComputedBuffer, ir.TemplateBuffer)), f'type(self.node)={type(self.node)!r}'
        return bool(self.node.get_reduction_type())

    def is_template(self):
        return isinstance(self.node, ir.TemplateBuffer)

    def run(self, *index_vars):
        self.decide_inplace_update()
        self.mark_run()
        self.codegen(index_vars)

    def mark_run(self):
        self.allocate()

    def ranges_from_index_vars(self, index_vars):
        sizes = self._sizes
        assert sum(map(len, sizes)) == sum(map(len, index_vars))
        var_ranges = dict(zip(itertools.chain.from_iterable(index_vars), itertools.chain.from_iterable(sizes)))
        return var_ranges

    def codegen(self, index_vars):
        var_ranges = self.ranges_from_index_vars(index_vars)
        try:
            with V.set_ops_handler(SimplifyIndexing(V.get_ops_handler(), var_ranges)), V.kernel.set_current_node(self):
                self._body(*index_vars)
        except Exception:
            log.fatal('Error in codegen for %s', self.node)
            raise

    def pointwise_read_writes(self):
        """
        Get the memory dependencies in the non-reduction axis.
        """
        sizes, reduction_sizes = self._sizes

        def fn(index):
            return self._body(index, [sympy.Integer(0) for _ in reduction_sizes])
        return dependencies.extract_read_writes(fn, sizes)

    def can_inplace(self, read_dep: dependencies.MemoryDep):
        if self.get_aliases() or self.is_template():
            return False
        if len(self.read_writes.writes) == 1 and isinstance(read_dep, dependencies.MemoryDep):
            write_dep = next(iter(self.read_writes.writes))
            assert isinstance(write_dep, dependencies.MemoryDep), f'type(write_dep)={type(write_dep)!r}'
            return read_dep.index == write_dep.index and read_dep.size == write_dep.size
        return False

    @cache_on_self
    def _get_atomic_add_buffers(self) -> Set[str]:
        buffers_store_as_atomic_add = set()
        if isinstance(self._body, ir.LoopBody):
            for node in self._body.get_nodes():
                if node.op == 'call_method' and node.target == 'store' and ('mode' in node.kwargs and node.kwargs['mode'] == 'atomic_add' or (len(node.args) == 5 and node.args[4] == 'atomic_add')):
                    buffers_store_as_atomic_add.add(node.kwargs['name'] if 'name' in node.kwargs else node.args[1] if len(node.args) >= 2 else '')
        return buffers_store_as_atomic_add

    def has_atomic_add(self, check_buf):
        return check_buf in self._get_atomic_add_buffers()