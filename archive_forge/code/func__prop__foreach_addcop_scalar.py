from typing import cast, List, Optional, Sequence, Tuple
import torch
from torch.distributed._tensor.op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.ops.common_rules import pointwise_rule
from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.placement_types import (
@register_prop_rule([aten._foreach_addcdiv.Scalar, aten._foreach_addcmul.Scalar])
def _prop__foreach_addcop_scalar(op_schema: OpSchema):
    self, tensor1, tensor2 = op_schema.args_schema[:3]
    scalar = None if len(op_schema.args_schema) < 4 else op_schema.args_schema[3]
    assert isinstance(self, list) and all((isinstance(s, DTensorSpec) for s in self))
    assert isinstance(tensor1, list) and all((isinstance(s, DTensorSpec) for s in self))
    assert isinstance(tensor2, list) and all((isinstance(s, DTensorSpec) for s in self))
    if any((s != t1 or s != t2 for s, t1, t2 in zip(self, tensor1, tensor2))):
        return OutputSharding(output_spec=None, schema_suggestions=[OpSchema(op=op_schema.op, args_schema=(self, self, self, scalar) if scalar else (self, self, self), kwargs_schema=op_schema.kwargs_schema)])
    else:
        return OutputSharding(output_spec=self)