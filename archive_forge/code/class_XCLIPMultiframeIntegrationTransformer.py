from copy import copy
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_x_clip import XCLIPConfig, XCLIPTextConfig, XCLIPVisionConfig
class XCLIPMultiframeIntegrationTransformer(nn.Module):
    """
    This corresponds to the `MultiframeIntegrationTransformer` class in the original implementation.
    """

    def __init__(self, config: XCLIPVisionConfig):
        super().__init__()
        self.position_embedding = nn.Parameter(torch.empty(1, config.num_frames, config.hidden_size))
        self.encoder = XCLIPEncoder(config)

    def forward(self, hidden_states, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutput]:
        residual = hidden_states
        hidden_states = hidden_states + self.position_embedding
        encoder_outputs = self.encoder(inputs_embeds=hidden_states, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = last_hidden_state.type(hidden_states.dtype) + residual
        pooled_output = last_hidden_state.mean(dim=1, keepdim=False)
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPooling(last_hidden_state=last_hidden_state, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)