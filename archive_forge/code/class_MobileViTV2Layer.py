from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_mobilevitv2 import MobileViTV2Config
class MobileViTV2Layer(nn.Module):
    """
    MobileViTV2 layer: https://arxiv.org/abs/2206.02680
    """

    def __init__(self, config: MobileViTV2Config, in_channels: int, out_channels: int, attn_unit_dim: int, n_attn_blocks: int=2, dilation: int=1, stride: int=2) -> None:
        super().__init__()
        self.patch_width = config.patch_size
        self.patch_height = config.patch_size
        cnn_out_dim = attn_unit_dim
        if stride == 2:
            self.downsampling_layer = MobileViTV2InvertedResidual(config, in_channels=in_channels, out_channels=out_channels, stride=stride if dilation == 1 else 1, dilation=dilation // 2 if dilation > 1 else 1)
            in_channels = out_channels
        else:
            self.downsampling_layer = None
        self.conv_kxk = MobileViTV2ConvLayer(config, in_channels=in_channels, out_channels=in_channels, kernel_size=config.conv_kernel_size, groups=in_channels)
        self.conv_1x1 = MobileViTV2ConvLayer(config, in_channels=in_channels, out_channels=cnn_out_dim, kernel_size=1, use_normalization=False, use_activation=False)
        self.transformer = MobileViTV2Transformer(config, d_model=attn_unit_dim, n_layers=n_attn_blocks)
        self.layernorm = nn.GroupNorm(num_groups=1, num_channels=attn_unit_dim, eps=config.layer_norm_eps)
        self.conv_projection = MobileViTV2ConvLayer(config, in_channels=cnn_out_dim, out_channels=in_channels, kernel_size=1, use_normalization=True, use_activation=False)

    def unfolding(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        batch_size, in_channels, img_height, img_width = feature_map.shape
        patches = nn.functional.unfold(feature_map, kernel_size=(self.patch_height, self.patch_width), stride=(self.patch_height, self.patch_width))
        patches = patches.reshape(batch_size, in_channels, self.patch_height * self.patch_width, -1)
        return (patches, (img_height, img_width))

    def folding(self, patches: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        batch_size, in_dim, patch_size, n_patches = patches.shape
        patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)
        feature_map = nn.functional.fold(patches, output_size=output_size, kernel_size=(self.patch_height, self.patch_width), stride=(self.patch_height, self.patch_width))
        return feature_map

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.downsampling_layer:
            features = self.downsampling_layer(features)
        features = self.conv_kxk(features)
        features = self.conv_1x1(features)
        patches, output_size = self.unfolding(features)
        patches = self.transformer(patches)
        patches = self.layernorm(patches)
        features = self.folding(patches, output_size)
        features = self.conv_projection(features)
        return features