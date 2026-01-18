import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, logging, replace_return_docstrings
from .configuration_fastspeech2_conformer import (
class FastSpeech2ConformerEncoderLayer(nn.Module):

    def __init__(self, config: FastSpeech2ConformerConfig, module_config):
        super().__init__()
        self.self_attn = FastSpeech2ConformerAttention(config, module_config)
        self.feed_forward = FastSpeech2ConformerMultiLayeredConv1d(config, module_config)
        self.macaron_style = config.use_macaron_style_in_conformer
        if self.macaron_style:
            self.feed_forward_macaron = FastSpeech2ConformerMultiLayeredConv1d(config, module_config)
            self.ff_macaron_layer_norm = nn.LayerNorm(config.hidden_size)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        self.use_cnn_module = config.use_cnn_in_conformer
        if self.use_cnn_module:
            self.conv_module = FastSpeech2ConformerConvolutionModule(config, module_config)
            self.conv_layer_norm = nn.LayerNorm(config.hidden_size)
            self.final_layer_norm = nn.LayerNorm(config.hidden_size)
        self.ff_layer_norm = nn.LayerNorm(config.hidden_size)
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(module_config['dropout_rate'])
        self.size = config.hidden_size
        self.normalize_before = module_config['normalize_before']
        self.concat_after = module_config['concat_after']
        if self.concat_after:
            self.concat_linear = nn.Linear(config.hidden_size + config.hidden_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, pos_emb: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, output_attentions: Optional[torch.Tensor]=False):
        """
        Compute encoded features.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch, time, size)`): Input tensor.
            pos_emb (`torch.Tensor` of shape `(1, time, size)`): Positional embeddings tensor.
            attention_mask (`torch.Tensor` of shape `(batch, time)`): Attention mask tensor for the input.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        Returns:
            `torch.Tensor`: Output tensor of shape `(batch, time, size)`.

        """
        if self.macaron_style:
            residual = hidden_states
            if self.normalize_before:
                hidden_states = self.ff_macaron_layer_norm(hidden_states)
            hidden_states = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(hidden_states))
            if not self.normalize_before:
                hidden_states = self.ff_macaron_layer_norm(hidden_states)
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        attention_output, attention_scores = self.self_attn(hidden_states, attention_mask=attention_mask, pos_emb=pos_emb, output_attentions=output_attentions)
        if self.concat_after:
            x_concat = torch.cat((hidden_states, attention_output), dim=-1)
            hidden_states = self.concat_linear(x_concat)
            hidden_states = residual + hidden_states
        else:
            hidden_states = self.dropout(attention_output)
            hidden_states = residual + hidden_states
        if not self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        if self.use_cnn_module:
            residual = hidden_states
            if self.normalize_before:
                hidden_states = self.conv_layer_norm(hidden_states)
            hidden_states = self.conv_module(hidden_states)
            hidden_states = self.dropout(hidden_states)
            hidden_states = residual + hidden_states
            if not self.normalize_before:
                hidden_states = self.conv_layer_norm(hidden_states)
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.ff_layer_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + self.ff_scale * hidden_states
        if not self.normalize_before:
            hidden_states = self.ff_layer_norm(hidden_states)
        if self.conv_module is not None:
            hidden_states = self.final_layer_norm(hidden_states)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attention_scores,)
        return outputs