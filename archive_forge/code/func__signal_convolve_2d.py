import math
from typing import Optional, Tuple, Union
import torch
from torch import Tensor, tensor
from torch.nn.functional import conv2d, pad
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.distributed import reduce
def _signal_convolve_2d(input_img: Tensor, kernel: Tensor) -> Tensor:
    """Applies 2D signal convolution to the input tensor with the given kernel."""
    left_padding = int(math.floor((kernel.size(3) - 1) / 2))
    right_padding = int(math.ceil((kernel.size(3) - 1) / 2))
    top_padding = int(math.floor((kernel.size(2) - 1) / 2))
    bottom_padding = int(math.ceil((kernel.size(2) - 1) / 2))
    padded = _symmetric_reflect_pad_2d(input_img, pad=(left_padding, right_padding, top_padding, bottom_padding))
    kernel = kernel.flip([2, 3])
    return conv2d(padded, kernel, stride=1, padding=0)