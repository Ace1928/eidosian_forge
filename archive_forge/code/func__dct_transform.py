from enum import Enum
from typing import Optional, Tuple
import torch
from torch import Tensor
def _dct_transform(dense: Tensor, dim: int) -> Tensor:
    """Should take a tensor and perform a Discrete Cosine Transform on the tensor.

    Args:
        dense (Tensor):
            Input dense tensor (no zeros).
        dim (int):
            Which dimension to transform.
    Returns:
        (Tensor):
            transformed dense tensor DCT components
    """
    raise NotImplementedError('Support for DCT has not been implemented yet!')