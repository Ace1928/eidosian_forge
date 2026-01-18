import copy
import operator
from typing import Any, cast, Dict, List, Optional, Sequence, Tuple
import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.distributed._tensor import DeviceMesh, distribute_tensor, DTensor
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.placement_types import (
from torch.distributed._tensor.redistribute import redistribute_local_tensor
from torch.distributed.tensor.parallel.style import ColwiseParallel, ParallelStyle
from torch.export import ExportedProgram
from torch.export.exported_program import ExportGraphSignature
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.node import Node
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.utils import _pytree as pytree
class TensorParallelTransformPass(PassBase):
    """
    This pass is responsible for transforming a single-device graph into a tensor parallel
    graph. It will mark the placement strategy of each node in the graph,
    partition the graph into distributed graph, then shard the parameters/buffers accordingly.
    """

    def __init__(self, rank: int, world_size: int, device_type: str, state_dict: Dict[str, torch.Tensor], graph_signature: ExportGraphSignature, parallel_strategies: Dict[str, ParallelStyle]) -> None:
        super().__init__()
        self.rank = rank
        self.mesh = DeviceMesh(device_type, torch.arange(world_size))
        self.state_dict: Dict[str, torch.Tensor] = state_dict
        self.graph_signature = graph_signature
        self.parallel_strategies = parallel_strategies

    def call(self, graph_module) -> PassResult:
        gm = copy.deepcopy(graph_module)
        parameter_placements = _generate_parameter_and_buffer_placements(list(self.state_dict.keys()), self.parallel_strategies)
        placement_strategies = _mark_sharding(gm, self.graph_signature, self.mesh, parameter_placements)
        _partitioner(gm)
        _shard_state_dict(self.state_dict, placement_strategies, self.graph_signature, self.mesh)
        return PassResult(gm, True)