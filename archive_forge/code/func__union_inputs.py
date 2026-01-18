import math
import ctypes
import copy
from .random import uniform
from .symbol import Symbol
from . import symbol
from ..base import _LIB, check_call
from ..base import SymbolHandle, _as_list
from ..attribute import AttrScope
def _union_inputs(*graphs):
    inputs = []
    locs = []
    input_id_to_loc = {}
    for graph in graphs:
        name_to_input_vars = {sym.name: sym for sym in inputs}
        name_to_cut_g_syms = {sym.list_outputs()[0]: sym for sym in _cut_subgraph(graph)}
        name_to_input_syms = {sym.name: sym for sym in _get_graph_inputs(graph)}
        input_locs = []
        for name in graph.list_inputs():
            assert name in name_to_input_syms
            if name in name_to_input_vars:
                sym = name_to_input_vars[name]
            elif name in name_to_cut_g_syms:
                sym = name_to_cut_g_syms[name]
            else:
                sym = copy.deepcopy(name_to_input_syms[name])
            if id(sym) in input_id_to_loc:
                loc = input_id_to_loc[id(sym)]
            else:
                loc = len(input_id_to_loc)
                inputs.append(sym)
                input_id_to_loc[id(sym)] = loc
            input_locs.append(loc)
        locs.append(input_locs)
    return (inputs, locs)