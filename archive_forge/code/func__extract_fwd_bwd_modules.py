from torch.fx.experimental.proxy_tensor import is_sym_node, py_sym_types
from torch.fx.experimental.sym_node import magic_methods, method_to_operator
from torch.fx.experimental.symbolic_shapes import (
import torch
import torch.fx as fx
import operator
import math
import torch.utils._pytree as pytree
import copy
import os
import itertools
import sympy
from collections import defaultdict
from torch.fx.passes import graph_drawer
from typing import List, Optional, Tuple, Union
from .compile_utils import fx_graph_cse, get_aten_target
from . import config
import functools
def _extract_fwd_bwd_modules(joint_module: fx.GraphModule, saved_values, saved_sym_nodes, *, num_fwd_outputs):
    fwd_outputs, bwd_outputs = _extract_fwd_bwd_outputs(joint_module, num_fwd_outputs=num_fwd_outputs)
    primal_inputs = list(filter(_is_primal, joint_module.graph.nodes))
    tangent_inputs = list(filter(_is_tangent, joint_module.graph.nodes))
    fwd_seed_offset_inputs = list(filter(_is_fwd_seed_offset, joint_module.graph.nodes))
    bwd_seed_offset_inputs = list(filter(_is_bwd_seed_offset, joint_module.graph.nodes))
    fwd_graph = _extract_graph_with_inputs_outputs(joint_module.graph, primal_inputs + fwd_seed_offset_inputs, fwd_outputs + saved_values + saved_sym_nodes)
    bwd_graph = _extract_graph_with_inputs_outputs(joint_module.graph, saved_sym_nodes + saved_values + tangent_inputs + bwd_seed_offset_inputs, bwd_outputs)
    for node in bwd_graph.nodes:
        if node.op == 'placeholder' and (not node.users):
            for saved_value in saved_values:
                if saved_value.name == node.name:
                    saved_values.remove(saved_value)
                    break
            for saved_sym in saved_sym_nodes:
                if saved_sym.name == node.name:
                    saved_sym_nodes.remove(saved_sym)
                    break
    saved_symbols: Set[sympy.Symbol] = set()
    saved_sym_nodes_binding = []
    saved_sym_nodes_derived = []
    for node in saved_sym_nodes:
        symbol = is_symbol_binding_fx_node(node)
        if symbol:
            saved_symbols.add(symbol)
            saved_sym_nodes_binding.append(node)
        else:
            saved_sym_nodes_derived.append(node)
    symbol_bindings = find_symbol_binding_fx_nodes(joint_module.graph)
    for node in itertools.chain(saved_sym_nodes_derived, saved_values, tangent_inputs):
        if 'val' not in node.meta:
            continue
        new_symbols = free_symbols(node.meta['val']) - saved_symbols
        for s in sorted(new_symbols, key=lambda s: s.name):
            if s not in symbol_bindings:
                continue
            saved_sym_nodes_binding.append(symbol_bindings[s])
        saved_symbols |= new_symbols
    saved_sym_nodes.clear()
    saved_sym_nodes.extend(saved_sym_nodes_binding + saved_sym_nodes_derived)
    fwd_graph = _extract_graph_with_inputs_outputs(joint_module.graph, primal_inputs + fwd_seed_offset_inputs, fwd_outputs + saved_values + saved_sym_nodes)
    bwd_graph = _extract_graph_with_inputs_outputs(joint_module.graph, saved_sym_nodes + saved_values + tangent_inputs + bwd_seed_offset_inputs, bwd_outputs)
    fwd_module = fx.GraphModule(joint_module, fwd_graph)
    bwd_module = fx.GraphModule(joint_module, bwd_graph)
    return (fwd_module, bwd_module)