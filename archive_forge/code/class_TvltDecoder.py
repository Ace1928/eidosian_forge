import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_tvlt import TvltConfig
class TvltDecoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        self.decoder_layers = nn.ModuleList([TvltLayer(decoder_config) for _ in range(config.decoder_num_hidden_layers)])
        self.layernorm = nn.LayerNorm(config.decoder_hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False
        self.config = config

    def forward(self, hidden_states, output_attentions=False, output_hidden_states=False, return_dict=True):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.decoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(layer_module.__call__, hidden_states, None, output_attentions)
            else:
                layer_outputs = layer_module(hidden_states, output_attentions=output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        logits = self.layernorm(hidden_states)
        if not return_dict:
            return tuple((v for v in [logits, all_hidden_states, all_self_attentions] if v is not None))
        return TvltDecoderOutput(logits=logits, hidden_states=all_hidden_states, attentions=all_self_attentions)