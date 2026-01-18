import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_seamless_m4t_v2 import SeamlessM4Tv2Config
@add_start_docstrings('Transformer speech encoder consisting of *config.speech_encoder_layers* conformer self attention layers.\n    Each layer is a [`SeamlessM4Tv2ConformerEncoderLayer`].', SEAMLESS_M4T_V2_START_DOCSTRING)
class SeamlessM4Tv2SpeechEncoder(SeamlessM4Tv2PreTrainedModel):
    main_input_name = 'input_features'

    def __init__(self, config: SeamlessM4Tv2Config):
        super().__init__(config)
        self.feature_projection = SeamlessM4Tv2ConformerFeatureProjection(config)
        self.encoder = SeamlessM4Tv2ConformerEncoder(config)
        self.intermediate_ffn = SeamlessM4Tv2ConformerFeedForward(config, act_fn='relu', dropout=0.0)
        self.adapter = SeamlessM4Tv2ConformerAdapter(config) if config.add_adapter else None
        self.inner_layer_norm = nn.LayerNorm(config.hidden_size)
        self.post_init()

    def forward(self, input_features: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **kwargs) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_features is None:
            raise ValueError('Both `input_features` and `inputs_embeds` are `None` in `SeamlessM4Tv2SpeechEncoder.forward`.\n                Make sure one of them is not `None`.')
        hidden_states = self.feature_projection(input_features)
        encoder_outputs = self.encoder(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = encoder_outputs[0]
        expanded_hidden_states = self.intermediate_ffn(hidden_states)
        hidden_states = hidden_states + 0.5 * expanded_hidden_states
        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states, attention_mask=attention_mask)
        hidden_states = self.inner_layer_norm(hidden_states)
        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]
        return Wav2Vec2BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)