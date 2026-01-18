import logging
import operator
from dataclasses import dataclass
from enum import auto, Enum
from functools import partial
from typing import Any, Callable, cast, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.distributed._spmd.experimental_ops
import torch.fx as fx
from torch.distributed._spmd.comm_tensor import _get_tracer
from torch.distributed._spmd.graph_utils import OP
from torch.distributed._spmd.log_utils import get_logger
from torch.distributed._tensor import DeviceMesh, DTensor
from torch.distributed._tensor.op_schema import OpSchema
from torch.distributed._tensor.placement_types import (
from torch.distributed._tensor.redistribute import redistribute_local_tensor
from torch.fx.experimental.proxy_tensor import make_fx, proxy_slot
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_map, tree_map_only, tree_unflatten
def _get_dtensor_dispatch_graph(node: fx.Node, node_to_obj: Dict[fx.Node, Any], *, force_make_fx: bool=False, default_mesh: Optional[DeviceMesh]=None) -> Optional[fx.GraphModule]:
    with torch.no_grad():
        args = tree_map(partial(_remap_arg, node_to_obj), node.args)
        kwargs = tree_map(partial(_remap_arg, node_to_obj), node.kwargs)
        op_overload = cast(torch._ops.OpOverload, node.target)
        if any((a.is_shard() for a in pytree.arg_tree_leaves(*args) if isinstance(a, DSymInt))):
            if op_overload in VIEW_SYM_INT_CONSUMERS:
                assert len(kwargs) == 0, f'Expect empty kwargs, but got {kwargs}'
                node_to_obj[node] = VIEW_SYM_INT_CONSUMERS[op_overload](node, args)
                return None
            elif op_overload in FACTORY_SYM_INT_CONSUMERS:
                assert default_mesh is not None, 'Requires default mesh for factory ops'
                node_to_obj[node] = FACTORY_SYM_INT_CONSUMERS[op_overload](node, args, kwargs, default_mesh)
                return None
            else:
                assert isinstance(logger, logging.Logger)
                logger.warning('Assuming using local_value from SymInt for %sis mathematically correct. Full args are %s.', op_overload, args)
        if node.target == aten.view.default:
            op_overload = aten.reshape.default
        args = tree_map(lambda a: a.local_value if isinstance(a, DSymInt) else a, args)
        kwargs = tree_map(lambda a: a.local_value if isinstance(a, DSymInt) else a, kwargs)
        if op_overload in FACTORY_OPS:
            node_to_obj[node] = FACTORY_OPS[op_overload](node, args, kwargs, default_mesh)
            return None
        dispatch = partial(_dispatch_with_local_tensors, op_overload, kwargs=kwargs, specs=args)
        gm = make_fx(dispatch, _allow_non_fake_inputs=False)(args)
        gm.graph.eliminate_dead_code()
        return gm