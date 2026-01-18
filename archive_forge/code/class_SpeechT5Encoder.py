import math
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, L1Loss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_speecht5 import SpeechT5Config, SpeechT5HifiGanConfig
class SpeechT5Encoder(SpeechT5PreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* layers. Each layer is a [`SpeechT5EncoderLayer`].
    """

    def __init__(self, config: SpeechT5Config):
        super().__init__(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layerdrop = config.encoder_layerdrop
        self.layers = nn.ModuleList([SpeechT5EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.embed_positions = SpeechT5RelativePositionalEncoding(config.hidden_size // config.encoder_attention_heads, config.encoder_max_relative_position)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(self, hidden_states: torch.FloatTensor, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutput]:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, feature_size)`):
                Features extracted from the speech or text input by the encoder prenet.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing convolution and attention on padding token indices. Mask values selected in
                `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if attention_mask is not None:
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        position_bias = self.embed_positions(hidden_states)
        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        if head_mask is not None:
            if head_mask.size()[0] != len(self.layers):
                raise ValueError(f'The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}.')
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            skip_the_layer = False
            if self.training:
                dropout_probability = torch.rand([])
                skip_the_layer = dropout_probability < self.layerdrop
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(encoder_layer.__call__, hidden_states, attention_mask, head_mask[idx] if head_mask is not None else None, position_bias, output_attentions)
                else:
                    layer_outputs = encoder_layer(hidden_states, attention_mask=attention_mask, position_bias=position_bias, layer_head_mask=head_mask[idx] if head_mask is not None else None, output_attentions=output_attentions)
                hidden_states = layer_outputs[0]
            if skip_the_layer:
                layer_outputs = (None, None)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions)