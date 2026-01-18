import warnings
from typing import List, Optional, Tuple, Union
import torch
from torch import nn, Tensor
from torch.nn import functional as F
def _get_conv1d_layer(in_channels: int, out_channels: int, kernel_size: int=1, stride: int=1, padding: Optional[Union[str, int, Tuple[int]]]=None, dilation: int=1, bias: bool=True, w_init_gain: str='linear') -> torch.nn.Conv1d:
    """1D convolution with xavier uniform initialization.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int, optional): Number of channels in the input image. (Default: ``1``)
        stride (int, optional): Number of channels in the input image. (Default: ``1``)
        padding (str, int or tuple, optional): Padding added to both sides of the input.
            (Default: dilation * (kernel_size - 1) / 2)
        dilation (int, optional): Number of channels in the input image. (Default: ``1``)
        w_init_gain (str, optional): Parameter passed to ``torch.nn.init.calculate_gain``
            for setting the gain parameter of ``xavier_uniform_``. (Default: ``linear``)

    Returns:
        (torch.nn.Conv1d): The corresponding Conv1D layer.
    """
    if padding is None:
        if kernel_size % 2 != 1:
            raise ValueError('kernel_size must be odd')
        padding = int(dilation * (kernel_size - 1) / 2)
    conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
    torch.nn.init.xavier_uniform_(conv1d.weight, gain=torch.nn.init.calculate_gain(w_init_gain))
    return conv1d