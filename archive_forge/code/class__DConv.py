import math
import typing as tp
from typing import Any, Dict, List, Optional
import torch
from torch import nn
from torch.nn import functional as F
class _DConv(torch.nn.Module):
    """
    New residual branches in each encoder layer.
    This alternates dilated convolutions, potentially with LSTMs and attention.
    Also before entering each residual branch, dimension is projected on a smaller subspace,
    e.g. of dim `channels // compress`.

    Args:
        channels (int): input/output channels for residual branch.
        compress (float, optional): amount of channel compression inside the branch. (default: 4)
        depth (int, optional): number of layers in the residual branch. Each layer has its own
            projection, and potentially LSTM and attention.(default: 2)
        init (float, optional): initial scale for LayerNorm. (default: 1e-4)
        norm_type (bool, optional): Norm type, either ``group_norm `` or ``none`` (Default: ``group_norm``)
        attn (bool, optional): use LocalAttention. (Default: ``False``)
        heads (int, optional): number of heads for the LocalAttention.  (default: 4)
        ndecay (int, optional): number of decay controls in the LocalAttention. (default: 4)
        lstm (bool, optional): use LSTM. (Default: ``False``)
        kernel_size (int, optional): kernel size for the (dilated) convolutions. (default: 3)
    """

    def __init__(self, channels: int, compress: float=4, depth: int=2, init: float=0.0001, norm_type: str='group_norm', attn: bool=False, heads: int=4, ndecay: int=4, lstm: bool=False, kernel_size: int=3):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError('Kernel size should not be divisible by 2')
        self.channels = channels
        self.compress = compress
        self.depth = abs(depth)
        dilate = depth > 0
        norm_fn: tp.Callable[[int], nn.Module]
        norm_fn = lambda d: nn.Identity()
        if norm_type == 'group_norm':
            norm_fn = lambda d: nn.GroupNorm(1, d)
        hidden = int(channels / compress)
        act = nn.GELU
        self.layers = nn.ModuleList([])
        for d in range(self.depth):
            dilation = pow(2, d) if dilate else 1
            padding = dilation * (kernel_size // 2)
            mods = [nn.Conv1d(channels, hidden, kernel_size, dilation=dilation, padding=padding), norm_fn(hidden), act(), nn.Conv1d(hidden, 2 * channels, 1), norm_fn(2 * channels), nn.GLU(1), _LayerScale(channels, init)]
            if attn:
                mods.insert(3, _LocalState(hidden, heads=heads, ndecay=ndecay))
            if lstm:
                mods.insert(3, _BLSTM(hidden, layers=2, skip=True))
            layer = nn.Sequential(*mods)
            self.layers.append(layer)

    def forward(self, x):
        """DConv forward call

        Args:
            x (torch.Tensor): input tensor for convolution

        Returns:
            Tensor
                Output after being run through layers.
        """
        for layer in self.layers:
            x = x + layer(x)
        return x