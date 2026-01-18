import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def _build_cfg(self, all_states):
    graph = CFGraph()
    for state in all_states:
        b = state.pc_initial
        graph.add_node(b)
    for state in all_states:
        for edge in state.outgoing_edges:
            graph.add_edge(state.pc_initial, edge.pc, 0)
    graph.set_entry_point(0)
    graph.process()
    self.cfgraph = graph