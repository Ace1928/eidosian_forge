import copy
import dataclasses
import functools
from typing import (
import torch
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from torch.fx._compatibility import compatibility
from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode
from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.infra.pass_manager import PassManager
from .graph_signature import (  # noqa: F401
def _get_updated_range_constraints(gm: torch.fx.GraphModule) -> 'Dict[sympy.Symbol, Any]':

    def get_shape_env(gm):
        vals = [node.meta['val'] for node in gm.graph.nodes if node.meta.get('val', None) is not None]
        from torch._guards import detect_fake_mode
        fake_mode = detect_fake_mode(vals)
        if fake_mode is not None:
            return fake_mode.shape_env
        for v in vals:
            if isinstance(v, torch.SymInt):
                return v.node.shape_env
    shape_env = get_shape_env(gm)
    if shape_env is None:
        return {}
    range_constraints = {k: v for k, v in shape_env.var_to_range.items() if k not in shape_env.replacements}
    return range_constraints