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
def _unlift(gm, inp_pos_to_param_buffer_name, in_spec, out_spec, state_dict, tensor_constants, buffers_to_mutate):
    count = 0
    buffer_name_to_node = {}
    for node in gm.graph.nodes:
        if node.op == 'placeholder':
            if count in inp_pos_to_param_buffer_name:
                with gm.graph.inserting_after(node):
                    getattr_node = gm.graph.get_attr(inp_pos_to_param_buffer_name[count])
                    node.replace_all_uses_with(getattr_node)
                    metadata = node.meta
                    gm.graph.erase_node(node)
                    getattr_node.meta = metadata
                    buffer_name_to_node[inp_pos_to_param_buffer_name[count]] = getattr_node
            count += 1
        if node.op == 'output':
            user_output_nodes = []
            for return_node in pytree.tree_flatten(node.args)[0]:
                return_node_name = return_node.name
                if return_node_name in buffers_to_mutate:
                    buffer_node_name = buffers_to_mutate[return_node_name].replace('.', '_')
                    assert buffer_node_name in buffer_name_to_node
                    buffer_node = buffer_name_to_node[buffer_node_name]
                    with gm.graph.inserting_before(node):
                        gm.graph.call_function(torch.ops.aten.copy_.default, (buffer_node, return_node))
                else:
                    user_output_nodes.append(return_node)
            with gm.graph.inserting_before(node):
                new_output = gm.graph.output(tuple(user_output_nodes))
                node.replace_all_uses_with(new_output)
                gm.graph.erase_node(node)
    gm.graph.lint()
    if in_spec.type == tuple and len(in_spec.children_specs) == 2 and (in_spec.children_specs[0].type == tuple) and (in_spec.children_specs[1].type == dict):
        num_args = len(in_spec.children_specs[0].children_specs) + len(in_spec.children_specs[1].children_specs)
    else:
        num_args = len(in_spec.children_specs)
    names = [f'arg_{i}' for i in range(num_args)]
    gm.graph._codegen = _PyTreeCodeGen(_PyTreeInfo(names, in_spec, out_spec))
    gm.recompile()
    for node in gm.graph.nodes:
        if node.op == 'call_function' and node.target == torch.ops.cond:
            pred, true_graph, false_graph, operands = node.args
            true_gm = getattr(gm, true_graph.name)
            false_gm = getattr(gm, false_graph.name)
            inp_pos_to_param_buffer_name_for_submod = {}
            real_operands = []
            for ix, operand in enumerate(operands):
                if operand.target in inp_pos_to_param_buffer_name.values():
                    inp_pos_to_param_buffer_name_for_submod[ix] = operand.target
                    if operand.target in state_dict:
                        value = state_dict[operand.target]
                    elif operand.target in tensor_constants:
                        value = tensor_constants[operand.target]
                    else:
                        raise RuntimeError('Unable to find value for ', operand.target)
                    true_gm.register_buffer(operand.target, value)
                    false_gm.register_buffer(operand.target, value)
                else:
                    real_operands.append(operand)
            node.args = (pred, true_graph, false_graph, real_operands)
            _, in_spec = pytree.tree_flatten(real_operands)
            _unlift(true_gm, inp_pos_to_param_buffer_name_for_submod, in_spec, None, state_dict, tensor_constants, buffers_to_mutate)
            _unlift(false_gm, inp_pos_to_param_buffer_name_for_submod, in_spec, None, state_dict, tensor_constants, buffers_to_mutate)
        if node.op == 'call_function' and node.target.__name__ == 'map_impl':
            body_graph, num_mapped, *operands = node.args
            body_gm = getattr(gm, body_graph.name)
            inp_pos_to_buffer_name_for_submod = {}
            real_operands = []
            state_dict_for_lookup = {key.replace('.', '_'): value for key, value in state_dict.items()}
            for ix, operand in enumerate(operands):
                if operand.target in inp_pos_to_param_buffer_name.values():
                    inp_pos_to_buffer_name_for_submod[ix] = operand.target
                    if operand.target in state_dict_for_lookup:
                        value = state_dict_for_lookup[operand.target]
                    elif operand.target in tensor_constants:
                        value = tensor_constants[operand.target]
                    else:
                        raise RuntimeError(f'Unable to find value for {operand.target}')
                    body_gm.register_buffer(operand.target, value)
                else:
                    real_operands.append(operand)
            node.args = (body_graph, num_mapped, *real_operands)
            _, in_spec = pytree.tree_flatten(real_operands)
            _unlift(body_gm, inp_pos_to_buffer_name_for_submod, in_spec, None, state_dict, tensor_constants, buffers_to_mutate)
    gm.graph.lint()
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm