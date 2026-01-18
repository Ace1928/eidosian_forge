import math
from typing import Optional, Tuple, Union
import torch
from torch import Tensor, tensor
from torch.nn.functional import conv2d, pad
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.distributed import reduce
def _symmetric_reflect_pad_2d(input_img: Tensor, pad: Union[int, Tuple[int, ...]]) -> Tensor:
    """Applies symmetric padding to the 2D image tensor input using ``reflect`` mode (d c b a | a b c d | d c b a)."""
    if isinstance(pad, int):
        pad = (pad, pad, pad, pad)
    if len(pad) != 4:
        raise ValueError(f'Expected padding to have length 4, but got {len(pad)}')
    left_pad = input_img[:, :, :, 0:pad[0]].flip(dims=[3])
    right_pad = input_img[:, :, :, -pad[1]:].flip(dims=[3])
    padded = torch.cat([left_pad, input_img, right_pad], dim=3)
    top_pad = padded[:, :, 0:pad[2], :].flip(dims=[2])
    bottom_pad = padded[:, :, -pad[3]:, :].flip(dims=[2])
    return torch.cat([top_pad, padded, bottom_pad], dim=2)