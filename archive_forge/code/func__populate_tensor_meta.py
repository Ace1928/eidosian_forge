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
def _populate_tensor_meta(node: Node, output_spec: OutputSpecType) -> None:
    """
    Util function to populate tensor meta of output_spec based on node metadata.
    """
    if isinstance(node.meta['val'], Sequence):
        assert isinstance(output_spec, Sequence)
        for spec, fake_tensor in zip(output_spec, node.meta['val']):
            assert spec is not None
            spec.tensor_meta = TensorMeta(shape=fake_tensor.shape, stride=fake_tensor.stride(), dtype=fake_tensor.dtype)
    else:
        assert isinstance(output_spec, DTensorSpec)
        output_spec.tensor_meta = TensorMeta(shape=node.meta['val'].shape, stride=node.meta['val'].stride(), dtype=node.meta['val'].dtype)