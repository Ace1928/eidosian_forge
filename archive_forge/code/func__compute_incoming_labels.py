import abc
from collections.abc import Mapping
from typing import TypeVar, Generic
from numba_rvsdg.core.datastructures.scfg import SCFG
from numba_rvsdg.core.datastructures.basic_block import (
def _compute_incoming_labels(graph: Mapping[str, BasicBlock]) -> dict[str, set[str]]:
    """Returns a backward mapping from destination blocks to their
    incoming blocks.
    """
    jump_table: dict[str, set[str]] = {}
    blk: BasicBlock
    for k in graph:
        jump_table[k] = set()
    for blk in graph.values():
        for dst in blk.jump_targets:
            if dst in jump_table:
                jump_table[dst].add(blk.name)
    return jump_table