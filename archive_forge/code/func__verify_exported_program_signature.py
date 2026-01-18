import inspect
import math
import operator
from collections.abc import Iterable
from typing import Any, Dict, final, List, Optional, Tuple, Type
import torch
from torch._ops import HigherOrderOperator, OpOverload
from torch._subclasses.fake_tensor import FakeTensor
from torch.export.exported_program import ExportedProgram
from torch.export.graph_signature import (
from torch.fx import GraphModule
from torch.fx.experimental.symbolic_shapes import SymBool, SymFloat, SymInt
def _verify_exported_program_signature(exported_program) -> None:
    gs = exported_program.graph_signature
    input_node_names = [node.name for node in exported_program.graph.nodes if node.op == 'placeholder']
    if len(input_node_names) != len(gs.input_specs):
        raise SpecViolationError(f'Number of graph inputs ({len(input_node_names)}) does not match number of inputs in the graph signature ({len(gs.user_inputs)})')
    for input_spec, node in zip(gs.input_specs, input_node_names):
        if isinstance(input_spec.arg, (TensorArgument, SymIntArgument)):
            if input_spec.arg.name != node:
                raise SpecViolationError(f'Input spec name {input_spec.arg.name} does not match node name {node}')
        if input_spec.kind == InputKind.USER_INPUT:
            continue
        elif input_spec.kind == InputKind.PARAMETER:
            if not isinstance(input_spec.arg, TensorArgument):
                raise SpecViolationError(f'Parameter {input_spec.name} is not a tensor argument. Found {input_spec.arg} instead.')
            if input_spec.target is None:
                raise SpecViolationError(f'InputSpec for {input_spec.name} has no target.')
            param = input_spec.target
            if param not in exported_program.state_dict:
                raise SpecViolationError(f'Parameter {param} is not in the state dict.')
            if not isinstance(exported_program.state_dict[param], torch.nn.Parameter):
                raise SpecViolationError(f'State dict entry for parameter {param} is not an instance of torch.nn.Parameter.')
        elif input_spec.kind == InputKind.BUFFER:
            if not isinstance(input_spec.arg, TensorArgument):
                raise SpecViolationError(f'Buffer {input_spec.name} is not a tensor argument. Found {input_spec.arg} instead.')
            if input_spec.target is None:
                raise SpecViolationError(f'InputSpec for {input_spec.name} has no target.')
            buffer = input_spec.target
            if buffer not in exported_program.state_dict:
                raise SpecViolationError(f'Buffer {buffer} is not in the state dict.')
        elif input_spec.kind == InputKind.CONSTANT_TENSOR:
            if not isinstance(input_spec.arg, TensorArgument):
                raise SpecViolationError(f'Constant tensor {input_spec.name} is not a tensor argument. Found {input_spec.arg} instead.')
            if input_spec.target is None:
                raise SpecViolationError(f'InputSpec for {input_spec.name} has no target.')
            tensor_const = input_spec.target
            if tensor_const not in exported_program.tensor_constants:
                raise SpecViolationError(f'Constant tensor {tensor_const} is not in the tensor constants dictionary.')
        else:
            raise SpecViolationError(f'Unknown InputKind {input_spec.kind}.')
    output_node = list(exported_program.graph.nodes)[-1]
    assert output_node.op == 'output'
    output_nodes = [arg.name for arg in output_node.args[0]]
    if len(output_nodes) != len(gs.output_specs):
        raise SpecViolationError(f'Number of output nodes {len(output_nodes)} is different Than the number of outputs specified by the graph signature: \nNumber of mutated buffers: {len(gs.buffers_to_mutate)}. \nNumber of user outputs: {len(gs.user_outputs)}. \n')
    end = len(gs.buffers_to_mutate) + len(gs.user_inputs_to_mutate)
    mutate_nodes: List[str] = output_nodes[:end]
    user_output_nodes = output_nodes[end:end + len(gs.user_outputs)]
    for mutation_node in mutate_nodes:
        if mutation_node in gs.buffers_to_mutate:
            if gs.buffers_to_mutate[mutation_node] not in gs.buffers:
                raise SpecViolationError(f'Buffer output {mutation_node} does not point to a buffer that exists. \nDict of buffers that are mutated, in order: {gs.buffers_to_mutate} \nBuffer nodes available: {gs.buffers} \n')
        elif mutation_node in gs.user_inputs_to_mutate:
            if gs.user_inputs_to_mutate[mutation_node] not in gs.user_inputs:
                raise SpecViolationError(f'User input output {mutation_node} does not point to a user input that exists. \nDict of user inputs that are mutated, in order: {gs.user_inputs_to_mutate} \nUser input nodes available: {gs.user_inputs} \n')
        else:
            raise SpecViolationError(f'Mutation node {mutation_node} is neither a buffer nor a user input. Buffers to mutate: {gs.buffers_to_mutate}, User inputs to mutate: {gs.user_inputs_to_mutate}')
    for user_output_node, user_output_name in zip(user_output_nodes, gs.user_outputs):
        if user_output_node != user_output_name:
            raise SpecViolationError(f"User output {user_output_node} is not in the correct order or is not found in the exported program's user_output list: {gs.user_outputs}. ")