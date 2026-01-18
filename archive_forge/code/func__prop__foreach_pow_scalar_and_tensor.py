from typing import cast, List, Optional, Sequence, Tuple
import torch
from torch.distributed._tensor.op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.ops.common_rules import pointwise_rule
from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.placement_types import (
@register_prop_rule([aten._foreach_pow.ScalarAndTensor])
def _prop__foreach_pow_scalar_and_tensor(op_schema: OpSchema):
    scala, exponent = op_schema.args_schema
    assert isinstance(exponent, list) and all((isinstance(s, DTensorSpec) for s in exponent))
    return OutputSharding(output_spec=exponent)