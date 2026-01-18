import math
import typing as tp
from typing import Any, Dict, List, Optional
import torch
from torch import nn
from torch.nn import functional as F
class _HEncLayer(torch.nn.Module):
    """Encoder layer. This used both by the time and the frequency branch.
    Args:
        chin (int): number of input channels.
        chout (int): number of output channels.
        kernel_size (int, optional): Kernel size for encoder (Default: 8)
        stride (int, optional): Stride for encoder layer (Default: 4)
        norm_groups (int, optional): number of groups for group norm. (Default: 4)
        empty (bool, optional): used to make a layer with just the first conv. this is used
            before merging the time and freq. branches. (Default: ``False``)
        freq (bool, optional): boolean for whether conv layer is for frequency domain (Default: ``True``)
        norm_type (string, optional): Norm type, either ``group_norm `` or ``none`` (Default: ``group_norm``)
        context (int, optional): context size for the 1x1 conv. (Default: 0)
        dconv_kw (Dict[str, Any] or None, optional): dictionary of kwargs for the DConv class. (Default: ``None``)
        pad (bool, optional): true to pad the input. Padding is done so that the output size is
            always the input size / stride. (Default: ``True``)
    """

    def __init__(self, chin: int, chout: int, kernel_size: int=8, stride: int=4, norm_groups: int=4, empty: bool=False, freq: bool=True, norm_type: str='group_norm', context: int=0, dconv_kw: Optional[Dict[str, Any]]=None, pad: bool=True):
        super().__init__()
        if dconv_kw is None:
            dconv_kw = {}
        norm_fn = lambda d: nn.Identity()
        if norm_type == 'group_norm':
            norm_fn = lambda d: nn.GroupNorm(norm_groups, d)
        pad_val = kernel_size // 4 if pad else 0
        klass = nn.Conv1d
        self.freq = freq
        self.kernel_size = kernel_size
        self.stride = stride
        self.empty = empty
        self.pad = pad_val
        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
            pad_val = [pad_val, 0]
            klass = nn.Conv2d
        self.conv = klass(chin, chout, kernel_size, stride, pad_val)
        self.norm1 = norm_fn(chout)
        if self.empty:
            self.rewrite = nn.Identity()
            self.norm2 = nn.Identity()
            self.dconv = nn.Identity()
        else:
            self.rewrite = klass(chout, 2 * chout, 1 + 2 * context, 1, context)
            self.norm2 = norm_fn(2 * chout)
            self.dconv = _DConv(chout, **dconv_kw)

    def forward(self, x: torch.Tensor, inject: Optional[torch.Tensor]=None) -> torch.Tensor:
        """Forward pass for encoding layer.

        Size depends on whether frequency or time

        Args:
            x (torch.Tensor): tensor input of shape `(B, C, F, T)` for frequency and shape
                `(B, C, T)` for time
            inject (torch.Tensor, optional): on last layer, combine frequency and time branches through inject param,
                same shape as x (default: ``None``)

        Returns:
            Tensor
                output tensor after encoder layer of shape `(B, C, F / stride, T)` for frequency
                    and shape `(B, C, ceil(T / stride))` for time
        """
        if not self.freq and x.dim() == 4:
            B, C, Fr, T = x.shape
            x = x.view(B, -1, T)
        if not self.freq:
            le = x.shape[-1]
            if not le % self.stride == 0:
                x = F.pad(x, (0, self.stride - le % self.stride))
        y = self.conv(x)
        if self.empty:
            return y
        if inject is not None:
            if inject.shape[-1] != y.shape[-1]:
                raise ValueError('Injection shapes do not align')
            if inject.dim() == 3 and y.dim() == 4:
                inject = inject[:, :, None]
            y = y + inject
        y = F.gelu(self.norm1(y))
        if self.freq:
            B, C, Fr, T = y.shape
            y = y.permute(0, 2, 1, 3).reshape(-1, C, T)
            y = self.dconv(y)
            y = y.view(B, Fr, C, T).permute(0, 2, 1, 3)
        else:
            y = self.dconv(y)
        z = self.norm2(self.rewrite(y))
        z = F.glu(z, dim=1)
        return z