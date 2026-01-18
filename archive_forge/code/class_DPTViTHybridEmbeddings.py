import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...file_utils import (
from ...modeling_outputs import BaseModelOutput, DepthEstimatorOutput, SemanticSegmenterOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import ModelOutput, logging
from ...utils.backbone_utils import load_backbone
from .configuration_dpt import DPTConfig
class DPTViTHybridEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config, feature_size=None):
        super().__init__()
        image_size, patch_size = (config.image_size, config.patch_size)
        num_channels, hidden_size = (config.num_channels, config.hidden_size)
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = image_size[1] // patch_size[1] * (image_size[0] // patch_size[0])
        self.backbone = load_backbone(config)
        feature_dim = self.backbone.channels[-1]
        if len(self.backbone.channels) != 3:
            raise ValueError(f'Expected backbone to have 3 output features, got {len(self.backbone.channels)}')
        self.residual_feature_map_index = [0, 1]
        if feature_size is None:
            feat_map_shape = config.backbone_featmap_shape
            feature_size = feat_map_shape[-2:]
            feature_dim = feat_map_shape[1]
        else:
            feature_size = feature_size if isinstance(feature_size, collections.abc.Iterable) else (feature_size, feature_size)
            feature_dim = self.backbone.channels[-1]
        self.image_size = image_size
        self.patch_size = patch_size[0]
        self.num_channels = num_channels
        self.projection = nn.Conv2d(feature_dim, hidden_size, kernel_size=1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))

    def _resize_pos_embed(self, posemb, grid_size_height, grid_size_width, start_index=1):
        posemb_tok = posemb[:, :start_index]
        posemb_grid = posemb[0, start_index:]
        old_grid_size = int(math.sqrt(len(posemb_grid)))
        posemb_grid = posemb_grid.reshape(1, old_grid_size, old_grid_size, -1).permute(0, 3, 1, 2)
        posemb_grid = nn.functional.interpolate(posemb_grid, size=(grid_size_height, grid_size_width), mode='bilinear')
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, grid_size_height * grid_size_width, -1)
        posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
        return posemb

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool=False, return_dict: bool=False) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError('Make sure that the channel dimension of the pixel values match with the one set in the configuration.')
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]}).")
        position_embeddings = self._resize_pos_embed(self.position_embeddings, height // self.patch_size, width // self.patch_size)
        backbone_output = self.backbone(pixel_values)
        features = backbone_output.feature_maps[-1]
        output_hidden_states = [backbone_output.feature_maps[index] for index in self.residual_feature_map_index]
        embeddings = self.projection(features).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = embeddings + position_embeddings
        if not return_dict:
            return (embeddings, output_hidden_states)
        return BaseModelOutputWithIntermediateActivations(last_hidden_states=embeddings, intermediate_activations=output_hidden_states)