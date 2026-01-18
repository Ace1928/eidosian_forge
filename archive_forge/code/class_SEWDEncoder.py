import math
import warnings
from collections.abc import Sequence
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, LayerNorm
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import BaseModelOutput, CausalLMOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import softmax_backward_data
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_sew_d import SEWDConfig
class SEWDEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_conv_embed = SEWDPositionalConvEmbedding(config)
        self.pool = nn.AvgPool1d(config.squeeze_factor, config.squeeze_factor)
        self.encoder = SEWDTransformerEncoder(config)
        self.upsample = SEWDUpsampling(config)
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.tensor, attention_mask: Optional[torch.Tensor]=None, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        max_encoder_length = hidden_states.shape[1] // self.config.squeeze_factor
        if attention_mask is None:
            attention_mask = torch.ones((hidden_states.shape[0], max_encoder_length), dtype=torch.long, device=hidden_states.device)
        else:
            hidden_states[~attention_mask.bool()] = 0.0
            input_lengths = attention_mask.long().sum(-1)
            output_lengths = input_lengths // self.config.squeeze_factor
            attention_ids = torch.arange(0, max_encoder_length, device=output_lengths.device).view(1, -1).expand(output_lengths.shape[0], -1)
            attention_mask = (attention_ids < output_lengths.view(-1, 1)).long()
        n_input_timesteps = hidden_states.shape[1]
        hidden_states = hidden_states.transpose(1, 2)
        position_embeddings = self.pos_conv_embed(hidden_states)
        pooled_hidden_states = self.pool(hidden_states)
        min_length = min(position_embeddings.size(-1), pooled_hidden_states.size(-1))
        hidden_states = pooled_hidden_states[..., :min_length] + position_embeddings[..., :min_length]
        hidden_states = hidden_states.transpose(1, 2)
        encoder_outputs = self.encoder(hidden_states, attention_mask, output_hidden_states, output_attentions)
        hidden_states = self.upsample(encoder_outputs.last_hidden_state)
        if hidden_states.shape[1] < n_input_timesteps:
            hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, n_input_timesteps - hidden_states.shape[1]))
        if not return_dict:
            return tuple((v for v in [hidden_states, encoder_outputs.hidden_states, encoder_outputs.attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)