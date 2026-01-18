import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ....activations import ACT2FN
from ....file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ....integrations.deepspeed import is_deepspeed_zero3_enabled
from ....modeling_attn_mask_utils import _prepare_4d_attention_mask
from ....modeling_outputs import BaseModelOutput, CausalLMOutput
from ....modeling_utils import (
from ....utils import logging
from .configuration_mctct import MCTCTConfig
class MCTCTEncoder(MCTCTPreTrainedModel):

    def __init__(self, config: MCTCTConfig):
        super().__init__(config)
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.layer_norm = MCTCTLayerNorm()
        self.conv = MCTCTConv1dSubsampler(config)
        self.layers = nn.ModuleList([MCTCTLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, input_features: torch.Tensor, attention_mask: torch.Tensor, head_mask: torch.Tensor, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_features = self.layer_norm(input_features)
        inputs_embeds = self.conv(input_features)
        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(inputs_embeds.shape[1], attention_mask)
        hidden_states = nn.functional.dropout(inputs_embeds, p=self.hidden_dropout_prob, training=self.training)
        if attention_mask is not None:
            attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        if head_mask is not None:
            if head_mask.size()[0] != len(self.layers):
                raise ValueError(f'The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}.')
        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            dropout_probability = torch.rand([])
            skip_the_layer = True if self.training and dropout_probability < self.config.layerdrop else False
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(encoder_layer.__call__, hidden_states, attention_mask, head_mask[idx] if head_mask is not None else None, output_attentions)
                else:
                    layer_outputs = encoder_layer(hidden_states=hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)
                hidden_states = layer_outputs[0]
            if skip_the_layer:
                layer_outputs = (None, None)
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, encoder_states, all_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)