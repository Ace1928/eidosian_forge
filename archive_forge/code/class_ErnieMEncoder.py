import math
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn, tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_ernie_m import ErnieMConfig
class ErnieMEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([ErnieMEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, input_embeds: torch.Tensor, attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, output_attentions: Optional[bool]=False, output_hidden_states: Optional[bool]=False, return_dict: Optional[bool]=True) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        hidden_states = () if output_hidden_states else None
        attentions = () if output_attentions else None
        output = input_embeds
        if output_hidden_states:
            hidden_states = hidden_states + (output,)
        for i, layer in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            output, opt_attn_weights = layer(hidden_states=output, attention_mask=attention_mask, head_mask=layer_head_mask, past_key_value=past_key_value)
            if output_hidden_states:
                hidden_states = hidden_states + (output,)
            if output_attentions:
                attentions = attentions + (opt_attn_weights,)
        last_hidden_state = output
        if not return_dict:
            return tuple((v for v in [last_hidden_state, hidden_states, attentions] if v is not None))
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=last_hidden_state, hidden_states=hidden_states, attentions=attentions)