from typing import cast, List, Optional, Sequence, Tuple
import torch
from torch.distributed._tensor.op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.ops.common_rules import pointwise_rule
from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.placement_types import (
@register_prop_rule(aten.native_layer_norm.default)
def _prop_native_layer_norm(op_schema: OpSchema) -> OutputSharding:
    input, normalized_shape, weight, bias, eps = op_schema.args_schema
    assert isinstance(input, DTensorSpec)
    assert isinstance(normalized_shape, (tuple, list))
    if weight is not None:
        assert isinstance(weight, DTensorSpec)
        assert all((isinstance(p, Replicate) for p in weight.placements))
    if bias is not None:
        assert isinstance(bias, DTensorSpec)
        assert all((isinstance(p, Replicate) for p in bias.placements))
    batch_ndim = len(input.shape) - len(normalized_shape)
    assert all((isinstance(p, Replicate) or (isinstance(p, Shard) and p.dim < batch_ndim,) for p in input.placements))
    stats_spec = DTensorSpec(mesh=input.mesh, placements=input.placements)
    return OutputSharding(output_spec=(input, stats_spec, stats_spec))