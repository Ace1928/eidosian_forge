from typing import cast, List, Optional, Sequence, Tuple
import torch
from torch.distributed._tensor.op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.ops.common_rules import pointwise_rule
from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.placement_types import (
@register_prop_rule(aten.nll_loss_backward.default)
def _prop_nll_loss_backward(op_schema: OpSchema) -> OutputSharding:
    grad_output, self = op_schema.args_schema[:2]
    assert isinstance(grad_output, DTensorSpec)
    assert isinstance(self, DTensorSpec)
    return OutputSharding(output_spec=self)