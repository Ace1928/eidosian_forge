from typing import Any, Optional, Sequence, Tuple, Type, Union
import torch
from . import (
from .attn_bias import (
from .common import (
from .dispatch import _dispatch_bw, _dispatch_fw, _ensure_op_supports_or_raise
def _memory_efficient_attention(inp: Inputs, op: Optional[AttentionOp]=None) -> torch.Tensor:
    if all((x.requires_grad is False for x in [inp.query, inp.key, inp.value])):
        return _memory_efficient_attention_forward(inp, op=op[0] if op is not None else None)
    output_shape = inp.normalize_bmhk()
    return _fMHA.apply(op, inp.query, inp.key, inp.value, inp.attn_bias, inp.p, inp.scale).reshape(output_shape)