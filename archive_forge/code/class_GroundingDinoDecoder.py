import copy
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from ...activations import ACT2FN
from ...file_utils import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import meshgrid
from ...utils import is_accelerate_available, is_ninja_available, logging
from ...utils.backbone_utils import load_backbone
from ..auto import AutoModel
from .configuration_grounding_dino import GroundingDinoConfig
class GroundingDinoDecoder(GroundingDinoPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`GroundingDinoDecoderLayer`].

    The decoder updates the query embeddings through multiple self-attention and cross-attention layers.

    Some tweaks for Grounding DINO:

    - `position_embeddings`, `reference_points`, `spatial_shapes` and `valid_ratios` are added to the forward pass.
    - it also returns a stack of intermediate outputs and reference points from all decoding layers.

    Args:
        config: GroundingDinoConfig
    """

    def __init__(self, config: GroundingDinoConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layer_norm = nn.LayerNorm(config.d_model, config.layer_norm_eps)
        self.layers = nn.ModuleList([GroundingDinoDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.reference_points_head = GroundingDinoMLPPredictionHead(config.query_dim // 2 * config.d_model, config.d_model, config.d_model, 2)
        self.gradient_checkpointing = False
        self.bbox_embed = None
        self.class_embed = None
        self.query_scale = None
        self.post_init()

    def forward(self, inputs_embeds, vision_encoder_hidden_states, vision_encoder_attention_mask=None, text_encoder_hidden_states=None, text_encoder_attention_mask=None, reference_points=None, spatial_shapes=None, level_start_index=None, valid_ratios=None, self_attn_mask=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        """
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
                The query embeddings that are passed into the decoder.
            vision_encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Last hidden state from encoder related to vision feature map.
            vision_encoder_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding pixel features. Mask values selected in `[0, 1]`:
                - 1 for pixel features that are real (i.e. **not masked**),
                - 0 for pixel features that are padding (i.e. **masked**).
            text_encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, text_seq_len, hidden_size)`):
                Last hidden state from encoder related to text features.
            text_encoder_attention_mask (`torch.Tensor` of shape `(batch_size, text_seq_len)`, *optional*):
                Mask to avoid performing attention on padding text features. Mask values selected in `[0, 1]`:
                - 0 for text features that are real (i.e. **not masked**),
                - 1 for text features that are padding (i.e. **masked**).
            reference_points (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)` is `as_two_stage` else `(batch_size, num_queries, 2)` or , *optional*):
                Reference point in range `[0, 1]`, top-left (0,0), bottom-right (1, 1), including padding area.
            spatial_shapes (`torch.FloatTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of the feature maps.
            level_start_index (`torch.LongTensor` of shape `(num_feature_levels)`, *optional*):
                Indexes for the start of each feature level. In range `[0, sequence_length]`.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`, *optional*):
                Ratio of valid area in each feature level.
            self_attn_mask (`torch.BoolTensor` of shape `(batch_size, text_seq_len)`):
                Masks to avoid performing self-attention between vision hidden state. Mask values selected in `[0, 1]`:
                - 1 for queries that are real (i.e. **not masked**),
                - 0 for queries that are padding (i.e. **masked**).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_attns = () if output_attentions else None
        all_cross_attns_vision = () if output_attentions and vision_encoder_hidden_states is not None else None
        all_cross_attns_text = () if output_attentions and text_encoder_hidden_states is not None else None
        intermediate = ()
        intermediate_reference_points = ()
        for idx, decoder_layer in enumerate(self.layers):
            num_coordinates = reference_points.shape[-1]
            if num_coordinates == 4:
                reference_points_input = reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            elif num_coordinates == 2:
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]
            else:
                raise ValueError('Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}')
            query_pos = get_sine_pos_embed(reference_points_input[:, :, 0, :], num_pos_feats=self.config.d_model // 2)
            query_pos = self.reference_points_head(query_pos)
            if output_hidden_states:
                all_hidden_states += (self.layer_norm(hidden_states),)
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)
                    return custom_forward
                layer_outputs = torch.utils.checkpoint.checkpoint(create_custom_forward(decoder_layer), hidden_states, query_pos, reference_points_input, spatial_shapes, level_start_index, vision_encoder_hidden_states, vision_encoder_attention_mask, text_encoder_hidden_states, text_encoder_attention_mask, self_attn_mask, None)
            else:
                layer_outputs = decoder_layer(hidden_states=hidden_states, position_embeddings=query_pos, reference_points=reference_points_input, spatial_shapes=spatial_shapes, level_start_index=level_start_index, vision_encoder_hidden_states=vision_encoder_hidden_states, vision_encoder_attention_mask=vision_encoder_attention_mask, text_encoder_hidden_states=text_encoder_hidden_states, text_encoder_attention_mask=text_encoder_attention_mask, self_attn_mask=self_attn_mask, output_attentions=output_attentions)
            hidden_states = layer_outputs[0]
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[idx](hidden_states)
                num_coordinates = reference_points.shape[-1]
                if num_coordinates == 4:
                    new_reference_points = tmp + torch.special.logit(reference_points, eps=1e-05)
                    new_reference_points = new_reference_points.sigmoid()
                elif num_coordinates == 2:
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + torch.special.logit(reference_points, eps=1e-05)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    raise ValueError(f'Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}')
                reference_points = new_reference_points.detach()
            intermediate += (self.layer_norm(hidden_states),)
            intermediate_reference_points += (reference_points,)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                if text_encoder_hidden_states is not None:
                    all_cross_attns_text += (layer_outputs[2],)
                if vision_encoder_hidden_states is not None:
                    all_cross_attns_vision += (layer_outputs[3],)
        intermediate = torch.stack(intermediate, dim=1)
        intermediate_reference_points = torch.stack(intermediate_reference_points, dim=1)
        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if output_attentions:
            all_attns += (all_self_attns, all_cross_attns_text, all_cross_attns_vision)
        if not return_dict:
            return tuple((v for v in [hidden_states, intermediate, intermediate_reference_points, all_hidden_states, all_attns] if v is not None))
        return GroundingDinoDecoderOutput(last_hidden_state=hidden_states, intermediate_hidden_states=intermediate, intermediate_reference_points=intermediate_reference_points, hidden_states=all_hidden_states, attentions=all_attns)