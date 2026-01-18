import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_vits import VitsConfig
class VitsEncoder(nn.Module):

    def __init__(self, config: VitsConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([VitsEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        self.layerdrop = config.layerdrop

    def forward(self, hidden_states: torch.FloatTensor, padding_mask: torch.FloatTensor, attention_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        if attention_mask is not None:
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)
        hidden_states = hidden_states * padding_mask
        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()
        for encoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            dropout_probability = np.random.uniform(0, 1)
            skip_the_layer = self.training and dropout_probability < self.layerdrop
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(encoder_layer.__call__, hidden_states, padding_mask, attention_mask, output_attentions)
                else:
                    layer_outputs = encoder_layer(hidden_states, attention_mask=attention_mask, padding_mask=padding_mask, output_attentions=output_attentions)
                hidden_states = layer_outputs[0]
            if skip_the_layer:
                layer_outputs = (None, None)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        hidden_states = hidden_states * padding_mask
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions)