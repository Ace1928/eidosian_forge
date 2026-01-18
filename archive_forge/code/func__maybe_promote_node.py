from __future__ import annotations
import abc
import dataclasses
import inspect
import logging
from types import ModuleType
from typing import Any, Callable, Mapping, Optional, Sequence, Set
import torch
import torch._ops
import torch.fx
import torch.fx.traceback as fx_traceback
from torch import _prims_common, _refs
from torch._prims_common import (
from torch._refs import linalg as _linalg_refs, nn as _nn_refs, special as _special_refs
from torch._refs.nn import functional as _functional_refs
from torch._subclasses import fake_tensor
from torch.fx.experimental import proxy_tensor
from torch.fx.node import Node  # noqa: F401
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import _pass, diagnostics, type_utils as fx_type_utils
from torch.utils import _python_dispatch, _pytree
@_beartype.beartype
def _maybe_promote_node(self, diagnostic: diagnostics.Diagnostic, node: torch.fx.Node, rule: TypePromotionRule) -> torch.fx.Node:
    """Promote node inputs and outputs according to type promotion rule."""
    args, kwargs = self.fetch_args_kwargs_from_env(node)
    type_promotion_info = rule.preview_type_promotion(args, kwargs)
    new_args = []
    new_kwargs = {}
    for i, arg in enumerate(node.args):
        new_args.append(self._maybe_promote_arg(diagnostic, node, arg, type_promotion_info.args_dtypes.get(i, None)))
    for name, arg in node.kwargs.items():
        new_kwargs[name] = self._maybe_promote_arg(diagnostic, node, arg, type_promotion_info.kwargs_dtypes.get(name, None))
    new_args = tuple(new_args)
    if node.args != new_args or node.kwargs != new_kwargs:
        diagnostic.message = f'Applied type promotion for {node}. '
        node.args = new_args
        node.kwargs = new_kwargs
        self._rerun_node_after_type_promotion(diagnostic, node, type_promotion_info.out_dtype)
    else:
        diagnostic.message = f'Type promotion not needed for {node}. '
    return node