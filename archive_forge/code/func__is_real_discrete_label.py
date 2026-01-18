from typing import Optional, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
def _is_real_discrete_label(x: Tensor) -> bool:
    """Check if tensor of labels is real and discrete."""
    if x.ndim != 1:
        raise ValueError(f'Expected arguments to be 1-d tensors but got {x.ndim}-d tensors.')
    return not (torch.is_floating_point(x) or torch.is_complex(x))