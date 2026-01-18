import os
from dataclasses import dataclass, replace, field, fields
import dis
import operator
from functools import reduce
from typing import (
from collections import ChainMap
from numba_rvsdg.core.datastructures.byte_flow import ByteFlow
from numba_rvsdg.core.datastructures.scfg import SCFG
from numba_rvsdg.core.datastructures.basic_block import (
from numba_rvsdg.rendering.rendering import ByteFlowRenderer
from numba_rvsdg.core.datastructures import block_names
from numba.core.utils import MutableSortedSet, MutableSortedMap
from .regionpasses import (
class PropagateStack(RegionVisitor[_psData]):

    def __init__(self, _debug: bool=False):
        super(PropagateStack, self).__init__()
        self._debug = _debug

    def debug_print(self, *args, **kwargs):
        if self._debug:
            print(*args, **kwargs)

    def visit_block(self, block: BasicBlock, data: _psData) -> _psData:
        if isinstance(block, DDGBlock):
            nin = len(block.in_stackvars)
            inherited = data[:len(data) - nin]
            self.debug_print('--- stack', data)
            self.debug_print('--- inherited stack', inherited)
            out_stackvars = block.out_stackvars.copy()
            counts = reversed(list(range(len(inherited) + len(out_stackvars))))
            out_stack = block.out_stackvars[::-1]
            out_data = tuple([f'tos.{i}' for i in counts])
            unused_names = list(out_data)
            for vs in reversed(out_stack):
                k = unused_names.pop()
                op = Op('stack.export', bc_inst=None)
                op.add_input('0', vs)
                block.exported_stackvars[k] = vs = op.add_output(k)
                block.out_vars[k] = vs
            for orig in reversed(inherited):
                k = unused_names.pop()
                import_op = Op('var.incoming', bc_inst=None)
                block.in_vars[orig] = imported_vs = import_op.add_output(orig)
                op = Op('stack.export', bc_inst=None)
                op.add_input('0', imported_vs)
                vs = op.add_output(k)
                block.exported_stackvars[k] = vs
                block.out_vars[k] = vs
            self.debug_print('---=', block.name, 'out stack', out_data)
            return out_data
        else:
            return data

    def visit_loop(self, region: RegionBlock, data: _psData) -> _psData:
        self.debug_print('---LOOP_ENTER', region.name)
        data = self.visit_linear(region, data)
        self.debug_print('---LOOP_END=', region.name, 'stack', data)
        return data

    def visit_switch(self, region: RegionBlock, data: _psData) -> _psData:
        self.debug_print('---SWITCH_ENTER', region.name)
        header = region.header
        data_at_head = self.visit_linear(region.subregion[header], data)
        data_for_branches = []
        for blk in region.subregion.graph.values():
            if blk.kind == 'branch':
                data_for_branches.append(self.visit_linear(blk, data_at_head))
        data_after_branches = max(data_for_branches, key=len)
        exiting = region.exiting
        data_at_tail = self.visit_linear(region.subregion[exiting], data_after_branches)
        self.debug_print('data_at_head', data_at_head)
        self.debug_print('data_for_branches', data_for_branches)
        self.debug_print('data_after_branches', data_after_branches)
        self.debug_print('data_at_tail', data_at_tail)
        self.debug_print('---SWITCH_END=', region.name, 'stack', data_at_tail)
        return data_at_tail

    def make_data(self) -> _psData:
        return ()