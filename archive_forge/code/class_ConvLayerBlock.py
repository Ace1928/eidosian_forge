import logging
from typing import List, Optional, Tuple
import torch
from torch import nn, Tensor
from torch.nn import Module, Parameter
from .wavlm_attention import WavLMSelfAttention
class ConvLayerBlock(Module):
    """Convolution unit of FeatureExtractor"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, bias: bool, layer_norm: Optional[Module]):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.layer_norm = layer_norm
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, bias=bias)

    def forward(self, x: Tensor, length: Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x (Tensor): Shape: ``[batch, in_channels, in_frame]``.
            length (Tensor or None, optional): Shape ``[batch, ]``.
        Returns:
            Tensor: Shape ``[batch, out_channels, out_frames]``.
            Optional[Tensor]: Shape ``[batch, ]``.
        """
        x = self.conv(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        x = nn.functional.gelu(x)
        if length is not None:
            length = torch.div(length - self.kernel_size, self.stride, rounding_mode='floor') + 1
            length = torch.max(torch.zeros_like(length), length)
        return (x, length)