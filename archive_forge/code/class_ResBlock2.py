from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
class ResBlock2(torch.nn.Module):
    """Residual block of type 2 for HiFiGAN Vocoder :cite:`NEURIPS2020_c5d73680`.
    Args:
        channels (int): Number of channels in the input features.
        kernel_size (int, optional): Kernel size for 1D convolutions. (Default: ``3``)
        dilation (tuple of 2 ``int``, optional): Dilations for each 1D convolution. (Default: ``(1, 3)``)
        lrelu_slope (float): Slope of leaky ReLUs in activations.
    """

    def __init__(self, channels: int, kernel_size: int=3, dilation: Tuple[int, int]=(1, 3), lrelu_slope: float=0.1):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList([Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0])), Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))])
        self.lrelu_slope = lrelu_slope

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (Tensor): input of shape ``(batch_size, channels, time_length)``.
        Returns:
            Tensor of the same shape as input.
        """
        for c in self.convs:
            xt = F.leaky_relu(x, self.lrelu_slope)
            xt = c(xt)
            x = xt + x
        return x