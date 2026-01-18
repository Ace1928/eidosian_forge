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
def _fetch_fake_args(self) -> Sequence[Optional[fake_tensor.FakeTensor]]:
    """Fetch fake args from fx graph.

        For each argument, try to fetch fake tensor from the matching placeholder node.
        """
    fake_args = []
    for node in self.module.graph.nodes:
        if node.op == 'placeholder':
            try:
                fake_tensor = _fake_tensor_from_node_val(node)
            except RuntimeError as e:
                if not node.users:
                    fake_tensor = None
                else:
                    raise RuntimeError('Cannot fetch symbolic fake args from fx graph. InsertTypePromotion pass needs to run with pre-existing fake args, Otherwise the pass will produce inaccurate dynamic shape. ') from e
            fake_args.append(fake_tensor)
    return fake_args