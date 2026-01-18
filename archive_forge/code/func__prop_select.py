from typing import cast, List, Optional, Sequence, Tuple
import torch
from torch.distributed._tensor.op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.ops.common_rules import pointwise_rule
from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.placement_types import (
@register_prop_rule(aten.select.int)
def _prop_select(op_schema: OpSchema) -> OutputSharding:
    tensor, dim = op_schema.args_schema[:2]
    assert isinstance(tensor, DTensorSpec)
    assert isinstance(dim, int)
    placements: Sequence[Placement] = tensor.placements
    assert all((not p.is_shard(dim) for p in placements)), 'DTensor does not support select on sharded dimension.'
    new_placements: List[Placement] = []
    for p in placements:
        if isinstance(p, Shard) and p.dim > dim:
            new_placements.append(Shard(p.dim - 1))
        else:
            new_placements.append(p)
    return OutputSharding(output_spec=DTensorSpec(mesh=tensor.mesh, placements=tuple(new_placements)))