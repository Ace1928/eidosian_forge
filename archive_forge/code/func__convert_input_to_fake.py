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
def _convert_input_to_fake(gm, args, kwargs):
    if len(args) == 0 and len(kwargs) == 0 and (len(dict(gm.named_parameters())) == 0) and (len(dict(gm.named_buffers())) == 0):
        return ([], {}, {}, None)
    fake_inps: List[torch.Tensor] = []
    fake_mode = None
    for node in gm.graph.nodes:
        if node.op == 'placeholder' and 'val' in node.meta:
            fake_val = node.meta['val']
            if fake_val is not None and isinstance(fake_val, torch.Tensor):
                fake_inps.append(fake_val)
    if (detected_fake_mode := detect_fake_mode(fake_inps)):
        fake_mode = detected_fake_mode
    assert fake_mode is not None, "Cannot find fake_mode attatched to the graph's placeholders."
    count = 0

    def convert_to_fake(x):
        nonlocal count
        val = fake_inps[count]
        count += 1
        return val
    fake_args = pytree.tree_map_only(torch.Tensor, convert_to_fake, args)
    fake_kwargs = pytree.tree_map_only(torch.Tensor, fake_mode.from_tensor, kwargs)
    fake_params_buffers = pytree.tree_map_only(torch.Tensor, functools.partial(fake_mode.from_tensor, static_shapes=True), {**dict(gm.named_parameters(remove_duplicate=False)), **dict(gm.named_buffers(remove_duplicate=False))})
    return (fake_args, fake_kwargs, fake_params_buffers, fake_mode)