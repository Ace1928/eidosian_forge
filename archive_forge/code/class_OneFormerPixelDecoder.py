import copy
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from .configuration_oneformer import OneFormerConfig
class OneFormerPixelDecoder(nn.Module):

    def __init__(self, config: OneFormerConfig, feature_channels):
        super().__init__()
        self.config = config
        self.position_embedding = OneFormerSinePositionEmbedding(num_pos_feats=config.conv_dim // 2, normalize=True)
        self.num_feature_levels = 3
        transformer_in_channels = feature_channels[-self.num_feature_levels:]
        self.transformer_feature_strides = config.strides[-self.num_feature_levels:]
        self.feature_channels = feature_channels
        self.level_embed = nn.Parameter(torch.Tensor(self.num_feature_levels, config.conv_dim))
        if self.num_feature_levels > 1:
            input_projections_list = []
            for in_channels in transformer_in_channels[::-1]:
                input_projections_list.append(nn.Sequential(nn.Conv2d(in_channels, config.conv_dim, kernel_size=1), nn.GroupNorm(32, config.conv_dim)))
            self.input_projections = nn.ModuleList(input_projections_list)
        else:
            self.input_projections = nn.ModuleList([nn.Sequential(nn.Conv2d(transformer_in_channels[-1], config.conv_dim, kernel_size=1), nn.GroupNorm(32, config.conv_dim))])
        self.encoder = OneFormerPixelDecoderEncoderOnly(config)
        self.mask_projection = nn.Conv2d(config.conv_dim, config.mask_dim, kernel_size=1, stride=1, padding=0)
        self.common_stride = config.common_stride
        stride = min(self.transformer_feature_strides)
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))
        lateral_convs = []
        output_convs = []
        for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]):
            lateral_conv = nn.Sequential(nn.Conv2d(in_channels, config.conv_dim, kernel_size=1, bias=False), nn.GroupNorm(32, config.conv_dim))
            output_conv = nn.Sequential(nn.Conv2d(config.conv_dim, config.conv_dim, kernel_size=3, stride=1, padding=1, bias=False), nn.GroupNorm(32, config.conv_dim), nn.ReLU())
            self.add_module('adapter_{}'.format(idx + 1), lateral_conv)
            self.add_module('layer_{}'.format(idx + 1), output_conv)
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

    def get_valid_ratio(self, mask, dtype=torch.float32):
        """Get the valid ratio of all feature maps."""
        _, height, width = mask.shape
        valid_height = torch.sum(~mask[:, :, 0], 1)
        valid_width = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_heigth = valid_height.to(dtype) / height
        valid_ratio_width = valid_width.to(dtype) / width
        valid_ratio = torch.stack([valid_ratio_width, valid_ratio_heigth], -1)
        return valid_ratio

    def forward(self, features, encoder_outputs=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        sources = []
        position_embeddings_list = []
        for level, source in enumerate(features[::-1][:self.num_feature_levels]):
            sources.append(self.input_projections[level](source))
            position_embeddings_list.append(self.position_embedding(source))
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in sources]
        source_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for level, (source, mask, pos_embed) in enumerate(zip(sources, masks, position_embeddings_list)):
            batch_size, num_channels, height, width = source.shape
            spatial_shape = (height, width)
            spatial_shapes.append(spatial_shape)
            source = source.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[level].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            source_flatten.append(source)
            mask_flatten.append(mask)
        source_flatten = torch.cat(source_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=source_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m, dtype=source_flatten.dtype) for m in masks], 1)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(inputs_embeds=source_flatten, attention_mask=mask_flatten, position_embeddings=lvl_pos_embed_flatten, spatial_shapes=spatial_shapes, level_start_index=level_start_index, valid_ratios=valid_ratios, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        y = encoder_outputs.last_hidden_state
        bs = y.shape[0]
        split_size_or_sections = [None] * self.num_feature_levels
        for i in range(self.num_feature_levels):
            if i < self.num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = torch.split(y, split_size_or_sections, dim=1)
        out = []
        multi_scale_features = []
        num_cur_levels = 0
        for i, z in enumerate(y):
            out.append(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))
        for idx, feats in enumerate(features[:self.num_fpn_levels][::-1]):
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cur_fpn = lateral_conv(feats)
            y = cur_fpn + nn.functional.interpolate(out[-1], size=cur_fpn.shape[-2:], mode='bilinear', align_corners=False)
            y = output_conv(y)
            out.append(y)
        for o in out:
            if num_cur_levels < self.num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1
        return OneFormerPixelDecoderOutput(mask_features=self.mask_projection(out[-1]), multi_scale_features=multi_scale_features, attentions=encoder_outputs.attentions)