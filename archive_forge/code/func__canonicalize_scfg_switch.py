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
def _canonicalize_scfg_switch(scfg: SCFG):
    """Introduce "switch" region to enclose "head", "branch", "tail" regions."""
    todos = set(scfg.graph)
    while todos:
        label = todos.pop()
        todos.discard(label)
        block = scfg[label]
        if isinstance(block, RegionBlock):
            if block.kind == 'head':
                brlabels = block.jump_targets
                branches = [scfg[brlabel] for brlabel in brlabels]
                tail_label_candidates = set()
                for br in branches:
                    tail_label_candidates.update(br.jump_targets)
                [taillabel] = tail_label_candidates
                tail = scfg[taillabel]
                switch_labels = {label, taillabel, *brlabels}
                subregion_graph = {k: scfg[k] for k in switch_labels}
                scfg.remove_blocks(switch_labels)
                subregion_scfg = SCFG(graph=subregion_graph, name_gen=scfg.name_gen)
                todos -= switch_labels
                new_label = scfg.name_gen.new_region_name('switch')
                new_region = RegionBlock(name=new_label, _jump_targets=tail._jump_targets, kind='switch', parent_region=block.parent_region, header=block.name, exiting=taillabel, subregion=subregion_scfg)
                scfg.graph[new_label] = new_region
                for incoming_label, incoming_blk in scfg.graph.items():
                    if incoming_label != new_label and label in incoming_blk.jump_targets:
                        repl = {label: new_label}
                        replblk = _repl_jump_targets(incoming_blk, repl)
                        scfg.graph[incoming_label] = replblk
                if block.parent_region.header not in scfg.graph:
                    block.parent_region.replace_header(new_label)
                if block.parent_region.exiting not in scfg.graph:
                    block.parent_region.replace_exiting(new_label)
                _canonicalize_scfg_switch(subregion_graph[label].subregion)
                for br in brlabels:
                    _canonicalize_scfg_switch(subregion_graph[br].subregion)
                _canonicalize_scfg_switch(subregion_graph[taillabel].subregion)
            elif block.kind == 'loop':
                _canonicalize_scfg_switch(block.subregion)