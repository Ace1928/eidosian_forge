from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.distributed as dist
import torch.utils._pytree as pytree
from torch._subclasses import FakeTensorMode
from torch.distributed._spmd.data_parallel import (
from torch.distributed._spmd.distribute import _convert_to_distributed, Schema
from torch.distributed._tensor import DeviceMesh, Placement, Replicate, Shard
from torch.fx import GraphModule
class DTensorExpandMode(ParallelMode):
    """
    The DTensor Expand mode. It's replicating the parameters and
    shard the inputs to represent DDP like behavior, it's currently
    a transitent mode before we move to the new data parallel expansion.
    """

    def __init__(self, custom_passes: Optional[Callable[[GraphModule], GraphModule]]=None):
        self._placements_override: Dict[int, List[Placement]] = {}
        if custom_passes is not None:
            self._gm_passes: Callable[[GraphModule], GraphModule] = custom_passes
        else:
            self._gm_passes = lambda gm: gm

    def partition(self, gm: GraphModule, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer], params_and_buffers: Dict[str, Any], named_states: Dict[str, Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> GraphModule:
        flat_args = pytree.arg_tree_leaves(*args, **kwargs)
        mesh = DeviceMesh('cuda', torch.arange(dist.get_world_size()).cuda())
        shard_schema: Schema = Schema(mesh=mesh, placements=[Shard(0)])
        replicate_schema: Schema = Schema(mesh=mesh, placements=[Replicate()])
        inps, schemas = ([], [])
        for p in pytree.tree_leaves(params_and_buffers):
            assert isinstance(p, torch.Tensor), f'expecting Tensor but got {type(p)}'
            inps.append(p)
            schemas.append(replicate_schema)
        for o in pytree.tree_leaves(named_states):
            if isinstance(o, torch.Tensor):
                inps.append(o)
                schemas.append(replicate_schema)
            else:
                inps.append(torch.empty(0))
                schemas.append(replicate_schema)
        for a in flat_args:
            if isinstance(a, torch.Tensor):
                inps.append(a)
                if id(a) in self._placements_override:
                    schemas.append(Schema(mesh=mesh, placements=self._placements_override[id(a)]))
                else:
                    schemas.append(shard_schema)
            else:
                inps.append(torch.empty(0))
                schemas.append(shard_schema)
        with FakeTensorMode(allow_non_fake_inputs=True):
            fake_inps = [torch.empty_like(inp) for inp in inps]
        return _convert_to_distributed(gm, fake_inps, schemas, default_mesh=mesh, _allow_partial=False)[0]

    def transform_and_compile(self, gm: GraphModule) -> GraphModule:
        """
        Transform and compile a distributed graph with a set of graph transformation
        and optimization passes for the dtensor fallback parallel mode.
        """
        return self._gm_passes(gm)