from typing import Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812
def _reflection_pad_2d(inputs: Tensor, pad: int, outer_pad: int=0) -> Tensor:
    """Apply reflection padding to the input image.

    Args:
        inputs: Input image
        pad: Number of pads
        outer_pad: Number of outer pads

    Return:
        Padded image

    """
    for dim in [2, 3]:
        inputs = _single_dimension_pad(inputs, dim, pad, outer_pad)
    return inputs