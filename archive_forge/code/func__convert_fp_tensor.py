from typing import Any, Mapping, Type, Union
import torch
from torch import Tensor
def _convert_fp_tensor(tensor: Tensor, dst_type: Union[str, torch.dtype]) -> Tensor:
    return tensor.to(dst_type) if torch.is_floating_point(tensor) else tensor