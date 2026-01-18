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
class FunnelEncoder(nn.Module):

    def __init__(self, config: FunnelConfig) -> None:
        super().__init__()
        self.config = config
        self.attention_structure = FunnelAttentionStructure(config)
        self.blocks = nn.ModuleList([nn.ModuleList([FunnelLayer(config, block_index) for _ in range(block_size)]) for block_index, block_size in enumerate(config.block_sizes)])

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True) -> Union[Tuple, BaseModelOutput]:
        attention_mask = attention_mask.type_as(inputs_embeds)
        attention_inputs = self.attention_structure.init_attention_inputs(inputs_embeds, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden = inputs_embeds
        all_hidden_states = (inputs_embeds,) if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for block_index, block in enumerate(self.blocks):
            pooling_flag = hidden.size(1) > (2 if self.config.separate_cls else 1)
            pooling_flag = pooling_flag and block_index > 0
            if pooling_flag:
                pooled_hidden, attention_inputs = self.attention_structure.pre_attention_pooling(hidden, attention_inputs)
            for layer_index, layer in enumerate(block):
                for repeat_index in range(self.config.block_repeats[block_index]):
                    do_pooling = repeat_index == 0 and layer_index == 0 and pooling_flag
                    if do_pooling:
                        query = pooled_hidden
                        key = value = hidden if self.config.pool_q_only else pooled_hidden
                    else:
                        query = key = value = hidden
                    layer_output = layer(query, key, value, attention_inputs, output_attentions=output_attentions)
                    hidden = layer_output[0]
                    if do_pooling:
                        attention_inputs = self.attention_structure.post_attention_pooling(attention_inputs)
                    if output_attentions:
                        all_attentions = all_attentions + layer_output[1:]
                    if output_hidden_states:
                        all_hidden_states = all_hidden_states + (hidden,)
        if not return_dict:
            return tuple((v for v in [hidden, all_hidden_states, all_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=hidden, hidden_states=all_hidden_states, attentions=all_attentions)