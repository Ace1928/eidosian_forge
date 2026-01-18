import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_led import LEDConfig
class LEDEncoder(LEDPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self-attention layers. Each layer is a
    [`LEDEncoderLayer`].

    Args:
        config: LEDConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: LEDConfig, embed_tokens: Optional[nn.Embedding]=None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_encoder_position_embeddings
        if isinstance(config.attention_window, int):
            if config.attention_window % 2 != 0:
                raise ValueError('`config.attention_window` has to be an even value')
            if config.attention_window <= 0:
                raise ValueError('`config.attention_window` has to be positive')
            config.attention_window = [config.attention_window] * config.num_hidden_layers
        elif len(config.attention_window) != config.num_hidden_layers:
            raise ValueError(f'`len(config.attention_window)` should equal `config.num_hidden_layers`. Expected {config.num_hidden_layers}, given {len(config.attention_window)}')
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)
        self.embed_positions = LEDLearnedPositionalEmbedding(self.max_source_positions, embed_dim)
        self.layers = nn.ModuleList([LEDEncoderLayer(config, i) for i in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)
        self.gradient_checkpointing = False
        self.post_init()

    def _merge_to_attention_mask(self, attention_mask: torch.Tensor, global_attention_mask: torch.Tensor):
        if attention_mask is not None:
            attention_mask = attention_mask * (global_attention_mask + 1)
        else:
            attention_mask = global_attention_mask + 1
        return attention_mask

    def _pad_to_window_size(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, inputs_embeds: torch.Tensor, pad_token_id: int):
        """A helper function to pad tokens and mask to work with implementation of Longformer self-attention."""
        attention_window = self.config.attention_window if isinstance(self.config.attention_window, int) else max(self.config.attention_window)
        if attention_window % 2 != 0:
            raise ValueError(f'`attention_window` should be an even value. Given {attention_window}')
        input_shape = input_ids.shape if input_ids is not None else inputs_embeds.shape
        batch_size, seq_len = input_shape[:2]
        padding_len = (attention_window - seq_len % attention_window) % attention_window
        if padding_len > 0:
            logger.warning_once(f'Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of `config.attention_window`: {attention_window}')
            if input_ids is not None:
                input_ids = nn.functional.pad(input_ids, (0, padding_len), value=pad_token_id)
            if inputs_embeds is not None:
                input_ids_padding = inputs_embeds.new_full((batch_size, padding_len), self.config.pad_token_id, dtype=torch.long)
                inputs_embeds_padding = self.embed_tokens(input_ids_padding)
                inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_padding], dim=-2)
            attention_mask = nn.functional.pad(attention_mask, (0, padding_len), value=False)
        return (padding_len, input_ids, attention_mask, inputs_embeds)

    def forward(self, input_ids=None, attention_mask=None, global_attention_mask=None, head_mask=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None):
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
            global_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to decide the attention given on each token, local attention or global attention for the encoder.
                Tokens with global attention attends to all other tokens, and all other tokens attend to them. This is
                important for task-specific finetuning because it makes the model more flexible at representing the
                task. For example, for classification, the <s> token should be given global attention. For QA, all
                question tokens should also have global attention. Please refer to the [Longformer
                paper](https://arxiv.org/abs/2004.05150) for more details. Mask values selected in `[0, 1]`:

                - 0 for local attention (a sliding window attention),
                - 1 for global attention (tokens that attend to all other tokens, and all other tokens attend to them).
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
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
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is None and inputs_embeds is None:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones(inputs_embeds.size()[:-1], device=inputs_embeds.device, dtype=torch.long)
        if global_attention_mask is not None:
            attention_mask = self._merge_to_attention_mask(attention_mask, global_attention_mask)
        padding_len, input_ids, attention_mask, inputs_embeds = self._pad_to_window_size(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, pad_token_id=self.config.pad_token_id)
        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        if attention_mask is not None:
            attention_mask = _prepare_4d_attention_mask_inverted(attention_mask, inputs_embeds.dtype)[:, 0, 0, :]
        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = is_index_global_attn.flatten().any().item()
        embed_pos = self.embed_positions(input_shape)
        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_global_attentions = () if output_attentions and is_global_attn else None
        if head_mask is not None:
            if head_mask.size()[0] != len(self.layers):
                raise ValueError(f'The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}.')
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            dropout_probability = torch.rand([])
            if self.training and dropout_probability < self.layerdrop:
                layer_outputs = (None, None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(encoder_layer.__call__, hidden_states, attention_mask, head_mask[idx] if head_mask is not None else None, is_index_masked, is_index_global_attn, is_global_attn, output_attentions)
                else:
                    layer_outputs = encoder_layer(hidden_states, attention_mask=attention_mask, layer_head_mask=head_mask[idx] if head_mask is not None else None, is_index_masked=is_index_masked, is_index_global_attn=is_index_global_attn, is_global_attn=is_global_attn, output_attentions=output_attentions)
                hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1].transpose(1, 2),)
                if is_global_attn:
                    all_global_attentions = all_global_attentions + (layer_outputs[2].transpose(2, 3),)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        if padding_len > 0:
            hidden_states = hidden_states[:, :-padding_len]
            if output_hidden_states:
                encoder_states = tuple([state[:, :-padding_len] for state in encoder_states])
            if output_attentions:
                all_attentions = tuple([state[:, :, :-padding_len, :] for state in all_attentions])
        if not return_dict:
            return tuple((v for v in [hidden_states, encoder_states, all_attentions, all_global_attentions] if v is not None))
        return LEDEncoderBaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions, global_attentions=all_global_attentions)