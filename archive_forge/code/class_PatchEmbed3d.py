from functools import partial
from typing import Any, Callable, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from ...transforms._presets import VideoClassification
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._meta import _KINETICS400_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
from ..swin_transformer import PatchMerging, SwinTransformerBlock
class PatchEmbed3d(nn.Module):
    """Video to Patch Embedding.

    Args:
        patch_size (List[int]): Patch token size.
        in_channels (int): Number of input channels. Default: 3
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size: List[int], in_channels: int=3, embed_dim: int=96, norm_layer: Optional[Callable[..., nn.Module]]=None) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.tuple_patch_size = (patch_size[0], patch_size[1], patch_size[2])
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=self.tuple_patch_size, stride=self.tuple_patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        _, _, t, h, w = x.size()
        pad_size = _compute_pad_size_3d((t, h, w), self.tuple_patch_size)
        x = F.pad(x, (0, pad_size[2], 0, pad_size[1], 0, pad_size[0]))
        x = self.proj(x)
        x = x.permute(0, 2, 3, 4, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x