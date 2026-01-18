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
def _tuplify_outputs(aot_export):

    def _aot_export_non_strict(mod, args, **kwargs):

        class Wrapper(torch.nn.Module):

            def __init__(self, mod):
                super().__init__()
                self._export_root = mod

            def forward(self, *args, **kwargs):
                nonlocal out_spec
                flat_outs, out_spec = pytree.tree_flatten(self._export_root(*args, **kwargs))
                return tuple(flat_outs)
        gm, sig = aot_export(Wrapper(mod), args, **kwargs)

        def strip_root(x):
            return x[len('_export_root.'):] if x.startswith('_export_root.') else x
        sig.parameters = pytree.tree_map(strip_root, sig.parameters)
        sig.buffers = pytree.tree_map(strip_root, sig.buffers)
        sig.inputs_to_buffers = pytree.tree_map(strip_root, sig.inputs_to_buffers)
        sig.inputs_to_parameters = pytree.tree_map(strip_root, sig.inputs_to_parameters)
        sig.buffers_to_mutate = pytree.tree_map(strip_root, sig.buffers_to_mutate)
        return (gm, sig)
    return _aot_export_non_strict