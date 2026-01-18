import copy
import dataclasses
import functools
import io
import json
import pathlib
import re
import sys
import types
import warnings
import weakref
import zipfile
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from unittest.mock import patch
import sympy
import torch
import torch._dynamo
import torch.fx
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from torch._decomp import core_aten_decompositions, get_decompositions
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.exc import UserError, UserErrorType
from torch._dynamo.source import ConstantSource
from torch._export.passes.collect_tracepoints_pass import CollectTracepointsPass
from torch._functorch.aot_autograd import aot_export_module, GraphSignature
from torch._functorch.eager_transforms import functionalize
from torch._guards import detect_fake_mode
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.export import _create_constraint, _Dim, Constraint
from torch.export.exported_program import (
from torch.export.graph_signature import (
from torch.fx import traceback as fx_traceback
from torch.fx._compatibility import compatibility
from torch.fx.experimental.proxy_tensor import make_fx, maybe_disable_fake_tensor_mode
from torch.fx.experimental.symbolic_shapes import (
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo
from torch.utils._sympy.value_ranges import ValueRangeError, ValueRanges
from .exported_program import (
from .passes.add_runtime_assertions_for_constraints_pass import (
from .passes.lift_constant_tensor_pass import lift_constant_tensor_pass
from .passes.remove_runtime_assertions import _RemoveRuntimeAssertionsPass
from .passes.replace_sym_size_ops_pass import _replace_sym_size_ops_pass
from .passes.replace_view_ops_with_view_copy_ops_pass import (
from .wrappers import _wrap_submodules
def _lift_buffers_to_user_inputs(gm: torch.fx.GraphModule, graph_signature: GraphSignature, user_input_names: List[str]) -> Dict[str, str]:
    assert len(graph_signature.user_inputs) == 0
    assert graph_signature.backward_signature is None
    names = set(user_input_names)
    placeholders = [node for node in gm.graph.nodes if node.op == 'placeholder']
    start = len(graph_signature.parameters)
    end = start + len(graph_signature.buffers)
    buffer_nodes = placeholders[start:end]
    last_placeholder_node = placeholders[-1] if len(placeholders) > 0 else None
    old_nodes: Dict[str, torch.fx.Node] = {}
    for node in buffer_nodes:
        buffer_name = graph_signature.inputs_to_buffers[node.name]
        if buffer_name not in names:
            continue
        old_nodes[buffer_name] = node
    replaces = {}
    new_node_names: Dict[str, str] = {}
    with gm.graph.inserting_after(last_placeholder_node):
        for name in reversed(user_input_names):
            new_node = gm.graph.placeholder(name)
            new_node.target = new_node.name
            new_node_names[name] = new_node.name
            if name in old_nodes:
                old_node = old_nodes[name]
                new_node.meta = copy.copy(old_node.meta)
                old_node.replace_all_uses_with(new_node)
                replaces[old_node.name] = new_node.name
    new_node_names = dict(reversed(new_node_names.items()))
    for old_node in old_nodes.values():
        gm.graph.erase_node(old_node)
    gm.recompile()
    graph_signature.buffers = [b for b in graph_signature.buffers if b not in names]
    graph_signature.inputs_to_buffers = {i: b for i, b in graph_signature.inputs_to_buffers.items() if b not in names}
    user_inputs_to_mutate = {o: b for o, b in graph_signature.buffers_to_mutate.items() if b in names}
    graph_signature.buffers_to_mutate = {o: b for o, b in graph_signature.buffers_to_mutate.items() if b not in names}
    graph_signature.user_inputs.extend(new_node_names.values())
    graph_signature.user_outputs = [replaces[o] if o in replaces else o for o in graph_signature.user_outputs]
    return user_inputs_to_mutate