import math
from functools import partial
from typing import Any, Callable, List, Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from ..ops.misc import MLP, Permute
from ..ops.stochastic_depth import StochasticDepth
from ..transforms._presets import ImageClassification, InterpolationMode
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
class SwinTransformer(nn.Module):
    """
    Implements Swin Transformer from the `"Swin Transformer: Hierarchical Vision Transformer using
    Shifted Windows" <https://arxiv.org/abs/2103.14030>`_ paper.
    Args:
        patch_size (List[int]): Patch size.
        embed_dim (int): Patch embedding dimension.
        depths (List(int)): Depth of each Swin Transformer layer.
        num_heads (List(int)): Number of attention heads in different layers.
        window_size (List[int]): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob (float): Stochastic depth rate. Default: 0.1.
        num_classes (int): Number of classes for classification head. Default: 1000.
        block (nn.Module, optional): SwinTransformer Block. Default: None.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        downsample_layer (nn.Module): Downsample layer (patch merging). Default: PatchMerging.
    """

    def __init__(self, patch_size: List[int], embed_dim: int, depths: List[int], num_heads: List[int], window_size: List[int], mlp_ratio: float=4.0, dropout: float=0.0, attention_dropout: float=0.0, stochastic_depth_prob: float=0.1, num_classes: int=1000, norm_layer: Optional[Callable[..., nn.Module]]=None, block: Optional[Callable[..., nn.Module]]=None, downsample_layer: Callable[..., nn.Module]=PatchMerging):
        super().__init__()
        _log_api_usage_once(self)
        self.num_classes = num_classes
        if block is None:
            block = SwinTransformerBlock
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-05)
        layers: List[nn.Module] = []
        layers.append(nn.Sequential(nn.Conv2d(3, embed_dim, kernel_size=(patch_size[0], patch_size[1]), stride=(patch_size[0], patch_size[1])), Permute([0, 2, 3, 1]), norm_layer(embed_dim)))
        total_stage_blocks = sum(depths)
        stage_block_id = 0
        for i_stage in range(len(depths)):
            stage: List[nn.Module] = []
            dim = embed_dim * 2 ** i_stage
            for i_layer in range(depths[i_stage]):
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                stage.append(block(dim, num_heads[i_stage], window_size=window_size, shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size], mlp_ratio=mlp_ratio, dropout=dropout, attention_dropout=attention_dropout, stochastic_depth_prob=sd_prob, norm_layer=norm_layer))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            if i_stage < len(depths) - 1:
                layers.append(downsample_layer(dim, norm_layer))
        self.features = nn.Sequential(*layers)
        num_features = embed_dim * 2 ** (len(depths) - 1)
        self.norm = norm_layer(num_features)
        self.permute = Permute([0, 3, 1, 2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.head = nn.Linear(num_features, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.head(x)
        return x