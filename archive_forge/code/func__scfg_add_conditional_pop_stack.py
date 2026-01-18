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
def _scfg_add_conditional_pop_stack(bcmap, scfg: SCFG):
    extra_records = {}
    for blk in scfg.graph.values():
        if isinstance(blk, PythonBytecodeBlock):
            last_inst = blk.get_instructions(bcmap)[-1]
            handler = HandleConditionalPop()
            res = handler.handle(last_inst)
            if res is not None:
                for br_index, instlist in enumerate(res.branch_instlists):
                    k = blk._jump_targets[br_index]
                    extra_records[k] = (blk.name, (br_index, instlist))

    def _replace_jump_targets(blk, idx, repl):
        return replace(blk, _jump_targets=tuple([repl if i == idx else jt for i, jt in enumerate(blk._jump_targets)]))
    for label, (parent_label, (br_index, instlist)) in extra_records.items():
        newlabel = scfg.name_gen.new_block_name('python.extrabasicblock')
        scfg.graph[parent_label] = _replace_jump_targets(scfg.graph[parent_label], br_index, newlabel)
        ebb = ExtraBasicBlock.make(newlabel, label, instlist)
        scfg.graph[newlabel] = ebb