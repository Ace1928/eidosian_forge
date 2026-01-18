import copy
from collections import defaultdict
import dataclasses
from typing import Dict, List, Optional, Tuple
import warnings
import sympy
import torch
import torch.fx
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.symbolic_shapes import SymInt
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo
from torch.utils._sympy.value_ranges import ValueRanges
from torch._export.passes.add_runtime_assertions_for_constraints_pass import (
from torch.export.graph_signature import (
from torch.export.exported_program import (
from .utils import _check_input_constraints_pre_hook
def _process_constraints(graph_module: torch.fx.GraphModule, num_lifted_params_buffers: int, example_inputs: List[torch.Tensor]) -> Tuple[Dict[sympy.Symbol, ValueRanges], List[Tuple[InputDim, InputDim]]]:
    """
    Process the constraints stored in the graph module to return something more readable.

    Args:
        graph_module (torch.fx.GraphModule): GraphModule returned from
            dynamo.export, which contains the "input_shape_constraints" and
            "inline_constraints" metadata

        example_inputs: Flattened list of example inputs used to export the graph module

    Returns:
        range_constraints (Dict[sympy.Symbol, ValueRanges]): Mapping of
            symbols (from SymInts) appearing in the fake tensors in
            node.meta["val"] to their range constraints, which are a tuple
            containing (lower, upper) constraints.

        equality_constraints (List[Tuple[InputDim, InputDim]]): List of tuples
            of (node, dim) to mark that these dimensions are equal.
    """
    input_shape_constraints = graph_module.meta.get('input_shape_constraints', [])
    inline_constraints = graph_module.meta.get('inline_constraints', [])
    tensor_id_to_nodes: Dict[int, List[str]] = defaultdict(list)
    placeholder_nodes: Dict[str, torch.fx.Node] = {}
    for i, node in enumerate(graph_module.graph.nodes):
        if node.op != 'placeholder':
            break
        if i >= num_lifted_params_buffers:
            example_input = example_inputs[i - num_lifted_params_buffers]
            tensor_id_to_nodes[id(example_input)].append(node.name)
            placeholder_nodes[node.name] = node
    equality_constraints: List[Tuple[InputDim, InputDim]] = []
    multi_range_constraints: Dict[InputDim, List[ValueRanges]] = defaultdict(list)
    for constraint in input_shape_constraints:
        for node in tensor_id_to_nodes[constraint['t_id']]:
            node_dim = InputDim(node, constraint['dim'])
            multi_range_constraints[node_dim].append(ValueRanges(constraint['min'], constraint['max']))
            if (shared := constraint.get('shared', None)):
                for other_node in tensor_id_to_nodes[shared['t_id']]:
                    other_node_dim = InputDim(other_node, shared['dim'])
                    equality_constraints.append((node_dim, other_node_dim))
    range_constraints: Dict[sympy.Symbol, ValueRanges] = {}
    range_constraints = {symbol: inline_constraints[symbol] for symbol in inline_constraints}
    for input_dim, multi_range_constraint in multi_range_constraints.items():
        min_vals = [rc.lower for rc in multi_range_constraint]
        max_vals = [rc.upper for rc in multi_range_constraint]
        min_val = max(min_vals)
        max_val = min(max_vals)
        assert min_val <= max_val
        val = placeholder_nodes[input_dim.input_name].meta['val']
        assert isinstance(val, FakeTensor)
        symint = val.shape[input_dim.dim]
        assert isinstance(symint, SymInt), f'Expected SymInt but got {symint}: {type(symint)}'
        symbol = symint.node._expr
        range_constraints[symbol] = ValueRanges(min_val, max_val)
    return (range_constraints, equality_constraints)