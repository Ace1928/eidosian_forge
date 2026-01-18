from typing import cast, List, Optional, Sequence, Tuple
import torch
from torch.distributed._tensor.op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.ops.common_rules import pointwise_rule
from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.placement_types import (
def _refine_sharding(op_schema: OpSchema, active_dim: Optional[int]) -> Sequence[Placement]:
    """Considers 2 first inputs of op_schema as having same shape, and returns suggested placement for a pointwise operation."""
    args_schema = []
    for s in op_schema.args_schema[:2]:
        assert isinstance(s, DTensorSpec) and s.tensor_meta is not None
        args_schema.append(DTensorSpec(mesh=s.mesh, placements=s.placements, tensor_meta=TensorMeta(shape=torch.Size(s.shape[0:active_dim] + (1,) + s.shape[active_dim + 1:]) if active_dim is not None else s.shape, stride=s.tensor_meta.stride, dtype=s.tensor_meta.dtype)))
    op_schema = OpSchema(op=op_schema.op, args_schema=args_schema, kwargs_schema={})
    output_sharding = pointwise_rule(op_schema, linearity=False)
    if output_sharding.output_spec:
        assert isinstance(output_sharding.output_spec, DTensorSpec)
        return output_sharding.output_spec.placements
    else:
        assert output_sharding.schema_suggestions is not None
        out_schema = output_sharding.schema_suggestions[0].args_schema[0]
        assert isinstance(out_schema, DTensorSpec)
        return tuple(out_schema.placements)