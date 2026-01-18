import copy
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...generation import GenerationConfig
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel, SequenceSummary
from ...pytorch_utils import Conv1D
from ...utils import (
from .configuration_clvp import (
class ClvpEncoder(ClvpPreTrainedModel):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`ClvpEncoderLayer`].

    Args:
        config: ClvpConfig
    """

    def __init__(self, config: ClvpConfig):
        super().__init__(config)
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_pos_emb = ClvpRotaryPositionalEmbedding(config) if config.use_rotary_embedding else None
        self.layers = nn.ModuleList([ClvpEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.sequence_summary = SequenceSummary(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.token_embedding

    def set_input_embeddings(self, value):
        self.token_embedding = value

    def forward(self, input_ids: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutput]:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                input embeddings for the model. This bypasses the model's internal embedding lookup matrix.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            position_ids (`torch.LongTensor`, *optional*):
                Denotes the position ids of `input_ids`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            inputs_embeds = self.token_embedding(input_ids)
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if attention_mask is not None:
            attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(input_shape[1], dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        rotary_pos_emb = self.rotary_pos_emb(inputs_embeds) if self.rotary_pos_emb is not None else None
        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(encoder_layer.__call__, hidden_states, rotary_pos_emb, attention_mask, position_ids)
            else:
                layer_outputs = encoder_layer(hidden_states, rotary_pos_emb, attention_mask, position_ids, output_attentions=output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        last_hidden_state = hidden_states
        last_hidden_state = self.final_layer_norm(last_hidden_state)
        pooled_output = self.sequence_summary(last_hidden_state)
        embeds = self.projection(pooled_output)
        if not return_dict:
            return tuple((v for v in [embeds, last_hidden_state, pooled_output, encoder_states, all_attentions] if v is not None))
        return ClvpEncoderOutput(embeds=embeds, last_hidden_state=last_hidden_state, pooler_output=pooled_output, hidden_states=encoder_states, attentions=all_attentions)