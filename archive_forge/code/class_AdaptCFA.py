import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
class AdaptCFA(object):
    """Adapt Flow to the old CFA class expected by Interpreter
    """

    def __init__(self, flow):
        self._flow = flow
        self._blocks = {}
        for offset, blockinfo in flow.block_infos.items():
            self._blocks[offset] = AdaptCFBlock(blockinfo, offset)
        backbone = self._flow.cfgraph.backbone()
        graph = flow.cfgraph
        backbone = graph.backbone()
        inloopblocks = set()
        for b in self.blocks.keys():
            if graph.in_loops(b):
                inloopblocks.add(b)
        self._backbone = backbone - inloopblocks

    @property
    def graph(self):
        return self._flow.cfgraph

    @property
    def backbone(self):
        return self._backbone

    @property
    def blocks(self):
        return self._blocks

    def iterliveblocks(self):
        for b in sorted(self.blocks):
            yield self.blocks[b]

    def dump(self):
        self._flow.cfgraph.dump()