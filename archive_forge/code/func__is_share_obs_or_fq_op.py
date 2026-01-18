from typing import Callable
import torch
from torch.ao.quantization.quantizer import (
from torch.fx import Node
def _is_share_obs_or_fq_op(op: Callable) -> bool:
    return op in [torch.ops.aten.hardtanh.default, torch.ops.aten.hardtanh_.default, torch.ops.aten.mean.default, torch.ops.aten.mean.dim, torch.ops.aten.permute.default, torch.ops.aten.permute_copy.default, torch.ops.aten.squeeze.dim, torch.ops.aten.squeeze_copy.dim, torch.ops.aten.adaptive_avg_pool2d.default, torch.ops.aten.view_copy.default, torch.ops.aten.view.default, torch.ops.aten.slice_copy.Tensor, torch.ops.aten.flatten.using_ints]