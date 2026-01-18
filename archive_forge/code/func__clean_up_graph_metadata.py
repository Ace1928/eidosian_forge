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
def _clean_up_graph_metadata(gm: torch.fx.GraphModule) -> None:
    """
    Clean up the graph by removing sharding and partitioning related metadata
    """
    for node in gm.graph.nodes:
        if 'sharding' in node.meta:
            del node.meta['sharding']
        if 'val' in node.meta and isinstance(node.meta['val'], torch.Tensor):
            local_tensor_meta = _extract_tensor_metadata(node.meta['val'])
            node.meta['tensor_meta'] = local_tensor_meta