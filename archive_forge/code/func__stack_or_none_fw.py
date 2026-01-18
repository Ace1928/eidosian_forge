from typing import List, Optional, Sequence, Tuple, Union
import torch
from .common import _get_storage_base
def _stack_or_none_fw(tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int) -> Optional[torch.Tensor]:
    strides = get_stack_strides(tensors, dim)
    if strides is not None:
        input_shape = list(tensors[0].shape)
        input_shape.insert(dim, len(tensors))
        return tensors[0].as_strided(input_shape, strides)
    return None