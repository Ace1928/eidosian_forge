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
def _generate_default_output_sharding(node: Node, mesh: DeviceMesh, op_schema: OpSchema) -> OutputSharding:
    """
    Util function to create a default output sharding that suggests Replicate placement for both args and outputs.
    """

    def update_arg_spec(arg_spec: DTensorSpec) -> DTensorSpec:
        return DTensorSpec(mesh=arg_spec.mesh, placements=(Replicate(),), tensor_meta=arg_spec.tensor_meta)
    new_op_schema = OpSchema(op=op_schema.op, args_schema=pytree.tree_map_only(DTensorSpec, update_arg_spec, op_schema.args_schema), kwargs_schema=op_schema.kwargs_schema)

    def create_output_spec(tensor: FakeTensor) -> DTensorSpec:
        return DTensorSpec(mesh=mesh, placements=(Replicate(),), tensor_meta=TensorMeta(shape=tensor.shape, stride=tensor.stride(), dtype=tensor.dtype))
    return OutputSharding(output_spec=pytree.tree_map_only(FakeTensor, create_output_spec, node.meta['val']), schema_suggestions=[new_op_schema], failed_reason=f'{node.op} does not have sharding strategy registered', needs_redistribute=True)