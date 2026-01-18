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
def _mark_tensor_parallel_shardings(gm: GraphModule, graph_signature: ExportGraphSignature, mesh: DeviceMesh, parameter_placements: Dict[str, Placement]) -> Dict[Node, PlacementStrategy]:
    """
    Mark the placement strategies of the parameter and buffer placeholder nodes.
    """
    placement_strategies: Dict[Node, PlacementStrategy] = {}
    num_params_and_buffers = len(graph_signature.inputs_to_parameters) + len(graph_signature.inputs_to_buffers)
    placeholder_idx: int = 0
    for node in gm.graph.nodes:
        if node.op == 'placeholder':
            if placeholder_idx < num_params_and_buffers:
                fqn: str = _get_input_node_fqn(node.name, graph_signature)
                placement: Placement = parameter_placements[fqn] if fqn in parameter_placements else Replicate()
                placement_strategies[node] = _create_placement_strategy(node, mesh, placements=(placement,))
                placeholder_idx += 1
            else:
                placement_strategies[node] = _create_placement_strategy(node, mesh, placements=(Replicate(),))
    return placement_strategies