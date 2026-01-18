from typing import cast, List, Optional, Sequence, Tuple
import torch
from torch.distributed._tensor.op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.ops.common_rules import pointwise_rule
from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.placement_types import (
@register_prop_rule([aten._foreach_neg.default, aten._foreach_reciprocal.default, aten._foreach_sqrt.default])
def _prop__foreach_unaop(op_schema: OpSchema) -> OutputSharding:
    self = op_schema.args_schema[0]
    assert isinstance(self, list) and all((isinstance(s, DTensorSpec) for s in self))
    return OutputSharding(output_spec=self)