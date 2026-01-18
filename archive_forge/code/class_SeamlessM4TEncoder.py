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
from .configuration_seamless_m4t import SeamlessM4TConfig
@add_start_docstrings('Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a [`SeamlessM4TEncoderLayer`].', SEAMLESS_M4T_START_DOCSTRING, "\n        embed_tokens (`nn.Embedding`, *optional*):\n            Input embedding\n        is_t2u_encoder (`bool`, *optional*, defaults to `False`):\n            indicates if it belongs to the text-to-units model, in which case it won't have input embeddings\n    ")
class SeamlessM4TEncoder(SeamlessM4TPreTrainedModel):

    def __init__(self, config: SeamlessM4TConfig, embed_tokens: Optional[nn.Embedding]=None, is_t2u_encoder: bool=False):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        self.padding_idx = config.pad_token_id
        embed_dim = config.hidden_size
        self.is_t2u_encoder = is_t2u_encoder
        self.max_source_positions = config.max_position_embeddings
        if not self.is_t2u_encoder:
            self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)
            if embed_tokens is not None:
                self.embed_tokens.weight = embed_tokens.weight
            self.embed_positions = SeamlessM4TSinusoidalPositionalEmbedding(self.max_source_positions, embed_dim, self.padding_idx)
        layers = []
        for _ in range(config.encoder_layers):
            layers.append(SeamlessM4TEncoderLayer(config, encoder_attention_heads=config.encoder_attention_heads, encoder_ffn_dim=config.encoder_ffn_dim))
        self.layers = nn.ModuleList(layers)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(self, input_ids: torch.LongTensor=None, attention_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **kwargs) -> Union[Tuple, BaseModelOutput]:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
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
        if input_ids is not None and self.is_t2u_encoder:
            raise ValueError('You cannot pass input_ids to the encoder of the text_to_units model. Pass inputs_embeds instead.')
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            input = input_ids
            input_shape = input.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        if not self.is_t2u_encoder:
            embed_pos = self.embed_positions(input)
            hidden_states = inputs_embeds + embed_pos.to(inputs_embeds.device)
        else:
            hidden_states = inputs_embeds
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        if attention_mask is not None:
            attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    to_drop = True
            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(encoder_layer.forward, hidden_states, attention_mask, output_attentions)
                else:
                    layer_outputs = encoder_layer(hidden_states, attention_mask, output_attentions=output_attentions)
                hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, encoder_states, all_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)