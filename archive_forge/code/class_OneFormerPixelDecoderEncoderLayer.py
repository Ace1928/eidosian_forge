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
class OneFormerPixelDecoderEncoderLayer(nn.Module):

    def __init__(self, config: OneFormerConfig):
        super().__init__()
        self.embed_dim = config.conv_dim
        self.self_attn = OneFormerPixelDecoderEncoderMultiscaleDeformableAttention(embed_dim=self.embed_dim, num_heads=config.num_attention_heads, n_levels=3, n_points=4)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.dropout = config.dropout
        self.activation_fn = nn.functional.relu
        self.activation_dropout = config.dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_feedforward_dim)
        self.fc2 = nn.Linear(config.encoder_feedforward_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.is_training = config.is_training

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, position_embeddings: torch.Tensor=None, reference_points=None, spatial_shapes=None, level_start_index=None, output_attentions: bool=False):
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Input to the layer.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Attention mask.
            position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings, to be added to `hidden_states`.
            reference_points (`torch.FloatTensor`, *optional*):
                Reference points.
            spatial_shapes (`torch.LongTensor`, *optional*):
                Spatial shapes of the backbone feature maps.
            level_start_index (`torch.LongTensor`, *optional*):
                Level start index.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states, attn_weights = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, encoder_hidden_states=hidden_states, encoder_attention_mask=attention_mask, position_embeddings=position_embeddings, reference_points=reference_points, spatial_shapes=spatial_shapes, level_start_index=level_start_index, output_attentions=output_attentions)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.is_training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.is_training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.is_training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        if self.is_training:
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs