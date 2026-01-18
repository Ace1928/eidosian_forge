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
def _construct_inp_pos_to_param_buffer_name(new_gm, graph_signature, state_dict, tensor_constants=None):
    param_buffer_name_to_corrected_name = {}
    for name, value in state_dict.items():
        if name in graph_signature.buffers:
            if '.' in name:
                new_gm.register_buffer(name.replace('.', '_'), value)
                param_buffer_name_to_corrected_name[name] = name.replace('.', '_')
            else:
                new_gm.register_buffer(name, value)
        if name in graph_signature.parameters:
            if '.' in name:
                new_gm.register_parameter(name.replace('.', '_'), value)
                param_buffer_name_to_corrected_name[name] = name.replace('.', '_')
            else:
                new_gm.register_parameter(name, value)
    if tensor_constants is not None and len(tensor_constants) > 0:
        assert hasattr(graph_signature, 'lifted_tensor_constants')
        for name, value in tensor_constants.items():
            if name in graph_signature.lifted_tensor_constants:
                new_gm.register_buffer(name, value)
                param_buffer_name_to_corrected_name[name] = name
    count = 0
    inp_pos_to_param_buffer_name = {}
    for node in new_gm.graph.nodes:
        if node.op == 'placeholder':
            if node.name in graph_signature.inputs_to_buffers:
                buffer_name = graph_signature.inputs_to_buffers[node.name]
                if buffer_name in param_buffer_name_to_corrected_name:
                    inp_pos_to_param_buffer_name[count] = param_buffer_name_to_corrected_name[buffer_name]
                else:
                    inp_pos_to_param_buffer_name[count] = buffer_name
            if node.name in graph_signature.inputs_to_parameters:
                param_name = graph_signature.inputs_to_parameters[node.name]
                if param_name in param_buffer_name_to_corrected_name:
                    inp_pos_to_param_buffer_name[count] = param_buffer_name_to_corrected_name[param_name]
                else:
                    inp_pos_to_param_buffer_name[count] = param_name
            if hasattr(graph_signature, 'inputs_to_lifted_tensor_constants'):
                if node.name in graph_signature.inputs_to_lifted_tensor_constants:
                    inp_pos_to_param_buffer_name[count] = graph_signature.inputs_to_lifted_tensor_constants[node.name]
            count += 1
    return inp_pos_to_param_buffer_name