from typing import cast, List, Optional, Sequence, Tuple
import torch
from torch.distributed._tensor.op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.ops.common_rules import pointwise_rule
from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.placement_types import (
@register_prop_rule(aten.native_layer_norm_backward.default)
def _prop_native_layer_norm_backward(op_schema: OpSchema) -> OutputSharding:
    grad, input, normalized_shape, result1, result2, weight, bias, grad_input_mask = op_schema.args_schema
    assert isinstance(grad, DTensorSpec)
    assert isinstance(grad_input_mask, (list, tuple))
    if weight is not None:
        assert isinstance(weight, DTensorSpec)
        assert all((isinstance(s, Replicate) for s in weight.placements))
    if bias is not None:
        assert isinstance(bias, DTensorSpec)
        assert all((isinstance(s, Replicate) for s in bias.placements))
    assert any((isinstance(s, Shard) and s.dim == 0 for s in grad.placements)), f'Got {grad.placements}'
    weight_grad = DTensorSpec(mesh=weight.mesh, placements=tuple([_Partial()] * weight.mesh.ndim)) if weight else None
    bias_grad = DTensorSpec(mesh=bias.mesh, placements=tuple([_Partial()] * bias.mesh.ndim)) if bias else None
    return OutputSharding(output_spec=(grad if grad_input_mask[0] else None, weight_grad if grad_input_mask[1] else None, bias_grad if grad_input_mask[2] else None))