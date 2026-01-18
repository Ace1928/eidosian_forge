import collections
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_timesformer import TimesformerConfig
class TimesformerLayer(nn.Module):

    def __init__(self, config: TimesformerConfig, layer_index: int) -> None:
        super().__init__()
        attention_type = config.attention_type
        drop_path_rates = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        drop_path_rate = drop_path_rates[layer_index]
        self.drop_path = TimeSformerDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.attention = TimeSformerAttention(config)
        self.intermediate = TimesformerIntermediate(config)
        self.output = TimesformerOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.config = config
        self.attention_type = attention_type
        if attention_type not in ['divided_space_time', 'space_only', 'joint_space_time']:
            raise ValueError('Unknown attention type: {}'.format(attention_type))
        if self.attention_type == 'divided_space_time':
            self.temporal_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.temporal_attention = TimeSformerAttention(config)
            self.temporal_dense = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, output_attentions: bool=False):
        num_frames = self.config.num_frames
        num_patch_width = self.config.image_size // self.config.patch_size
        batch_size = hidden_states.shape[0]
        num_spatial_tokens = (hidden_states.size(1) - 1) // num_frames
        num_patch_height = num_spatial_tokens // num_patch_width
        if self.attention_type in ['space_only', 'joint_space_time']:
            self_attention_outputs = self.attention(self.layernorm_before(hidden_states), output_attentions=output_attentions)
            attention_output = self_attention_outputs[0]
            outputs = self_attention_outputs[1:]
            hidden_states = hidden_states + self.drop_path(attention_output)
            layer_output = self.layernorm_after(hidden_states)
            layer_output = self.intermediate(layer_output)
            layer_output = self.output(layer_output)
            layer_output = hidden_states + self.drop_path(layer_output)
            outputs = (layer_output,) + outputs
            return outputs
        elif self.attention_type == 'divided_space_time':
            temporal_embedding = hidden_states[:, 1:, :]
            temporal_embedding = temporal_embedding.reshape(batch_size, num_patch_height, num_patch_width, num_frames, temporal_embedding.shape[2]).reshape(batch_size * num_patch_height * num_patch_width, num_frames, temporal_embedding.shape[2])
            temporal_attention_outputs = self.temporal_attention(self.temporal_layernorm(temporal_embedding))
            attention_output = temporal_attention_outputs[0]
            residual_temporal = self.drop_path(attention_output)
            residual_temporal = residual_temporal.reshape(batch_size, num_patch_height, num_patch_width, num_frames, residual_temporal.shape[2]).reshape(batch_size, num_patch_height * num_patch_width * num_frames, residual_temporal.shape[2])
            residual_temporal = self.temporal_dense(residual_temporal)
            temporal_embedding = hidden_states[:, 1:, :] + residual_temporal
            init_cls_token = hidden_states[:, 0, :].unsqueeze(1)
            cls_token = init_cls_token.repeat(1, num_frames, 1)
            cls_token = cls_token.reshape(batch_size * num_frames, 1, cls_token.shape[2])
            spatial_embedding = temporal_embedding
            spatial_embedding = spatial_embedding.reshape(batch_size, num_patch_height, num_patch_width, num_frames, spatial_embedding.shape[2]).permute(0, 3, 1, 2, 4).reshape(batch_size * num_frames, num_patch_height * num_patch_width, spatial_embedding.shape[2])
            spatial_embedding = torch.cat((cls_token, spatial_embedding), 1)
            spatial_attention_outputs = self.attention(self.layernorm_before(spatial_embedding), output_attentions=output_attentions)
            attention_output = spatial_attention_outputs[0]
            outputs = spatial_attention_outputs[1:]
            residual_spatial = self.drop_path(attention_output)
            cls_token = residual_spatial[:, 0, :]
            cls_token = cls_token.reshape(batch_size, num_frames, cls_token.shape[1])
            cls_token = torch.mean(cls_token, 1, True)
            residual_spatial = residual_spatial[:, 1:, :]
            residual_spatial = residual_spatial.reshape(batch_size, num_frames, num_patch_height, num_patch_width, residual_spatial.shape[2]).permute(0, 2, 3, 1, 4).reshape(batch_size, num_patch_height * num_patch_width * num_frames, residual_spatial.shape[2])
            residual = residual_spatial
            hidden_states = temporal_embedding
            hidden_states = torch.cat((init_cls_token, hidden_states), 1) + torch.cat((cls_token, residual), 1)
            layer_output = self.layernorm_after(hidden_states)
            layer_output = self.intermediate(layer_output)
            layer_output = self.output(layer_output)
            layer_output = hidden_states + self.drop_path(layer_output)
            outputs = (layer_output,) + outputs
            return outputs