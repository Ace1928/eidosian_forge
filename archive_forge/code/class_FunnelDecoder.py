import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_funnel import FunnelConfig
class FunnelDecoder(nn.Module):

    def __init__(self, config: FunnelConfig) -> None:
        super().__init__()
        self.config = config
        self.attention_structure = FunnelAttentionStructure(config)
        self.layers = nn.ModuleList([FunnelLayer(config, 0) for _ in range(config.num_decoder_layers)])

    def forward(self, final_hidden: torch.Tensor, first_block_hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True) -> Union[Tuple, BaseModelOutput]:
        upsampled_hidden = upsample(final_hidden, stride=2 ** (len(self.config.block_sizes) - 1), target_len=first_block_hidden.shape[1], separate_cls=self.config.separate_cls, truncate_seq=self.config.truncate_seq)
        hidden = upsampled_hidden + first_block_hidden
        all_hidden_states = (hidden,) if output_hidden_states else None
        all_attentions = () if output_attentions else None
        attention_inputs = self.attention_structure.init_attention_inputs(hidden, attention_mask=attention_mask, token_type_ids=token_type_ids)
        for layer in self.layers:
            layer_output = layer(hidden, hidden, hidden, attention_inputs, output_attentions=output_attentions)
            hidden = layer_output[0]
            if output_attentions:
                all_attentions = all_attentions + layer_output[1:]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden,)
        if not return_dict:
            return tuple((v for v in [hidden, all_hidden_states, all_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=hidden, hidden_states=all_hidden_states, attentions=all_attentions)