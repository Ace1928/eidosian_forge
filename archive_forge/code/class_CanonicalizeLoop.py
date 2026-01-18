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
class CanonicalizeLoop(RegionTransformer[None]):
    """
    Make sure loops has non-region header.

    Preferably, the exiting block should be non-region as well but
    it's hard to do with the current numba_rvsdg API.

    Doing this so we don't have to fixup backedges as backedges will always
    point to a non-region node in ``_canonicalize_scfg_switch``.
    """

    def visit_loop(self, parent: SCFG, region: RegionBlock, data: None):
        new_label = parent.name_gen.new_block_name(block_names.SYNTH_FILL)
        region.subregion.insert_SyntheticFill(new_label, {}, {region.header})
        region.replace_header(new_label)

        def get_inner_most_exiting(blk):
            while isinstance(blk, RegionBlock):
                parent, blk = (blk, blk.subregion.graph[blk.exiting])
            return (parent, blk)
        tail_parent, tail_bb = get_inner_most_exiting(region)
        [backedge] = tail_bb.backedges
        repl = {backedge: new_label}
        new_tail_bb = replace(tail_bb, backedges=(new_label,), _jump_targets=tuple([repl.get(x, x) for x in tail_bb._jump_targets]))
        tail_parent.subregion.graph[tail_bb.name] = new_tail_bb
        self.visit_linear(parent, region, data)

    def visit_block(self, parent: SCFG, block: BasicBlock, data: None):
        pass

    def visit_switch(self, parent: SCFG, block: BasicBlock, data: None):
        pass