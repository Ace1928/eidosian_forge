import dataclasses
import math
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_pegasus_x import PegasusXConfig
class PegasusXEncoder(PegasusXPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`PegasusXEncoderLayer`].

    Args:
        config: PegasusXConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: PegasusXConfig, embed_tokens: Optional[nn.Embedding]=None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        embed_dim = config.d_model
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim)
        self.embed_global = nn.Embedding(config.num_global_tokens, embed_dim)
        self.embed_positions = PegasusXSinusoidalPositionalEmbedding(embed_dim)
        self.layers = nn.ModuleList([PegasusXEncoderLayer(stagger_blocks_this_layer=i % 2 == 1 and config.stagger_local_blocks, config=config) for i in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.gradient_checkpointing = False
        self.post_init()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings matrix of the model if `new_num_position_embeddings !=
        config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embeddings. If position embeddings are learned, increasing the size will add
                newly initialized vectors at the end, whereas reducing the size will remove vectors from the end. If
                position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the size will
                add correct vectors at the end following the position encoding algorithm, whereas reducing the size
                will remove vectors from the end.
        """
        logger.info(f'Setting `config.max_position_embeddings={new_num_position_embeddings}`...')
        self.config.max_position_embeddings = new_num_position_embeddings
        self.embed_positions = PegasusXSinusoidalPositionalEmbedding(self.config.d_model)
        self.embed_positions.to(self.device)

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings matrix
        """
        return self.embed_positions

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None):
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
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        embed_pos = self.embed_positions(inputs_embeds)
        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        batch_size, seq_len, _ = hidden_states.shape
        if attention_mask is None:
            attention_mask = torch.ones(*input_shape, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        attention_mask = attention_mask.to(dtype=hidden_states.dtype)
        mask_min_value = torch.finfo(hidden_states.dtype).min
        inverted_mask = 1.0 - attention_mask
        attention_mask = inverted_mask.masked_fill(inverted_mask.to(torch.bool), mask_min_value)
        if seq_len % self.config.block_size != 0:
            pad_len = self.config.block_size - seq_len % self.config.block_size
            hidden_states = nn.functional.pad(hidden_states, pad=(0, 0, 0, pad_len), value=0)
            attention_mask = nn.functional.pad(attention_mask, pad=(0, pad_len), value=mask_min_value)
        global_hidden_states = self.embed_global(torch.arange(self.config.num_global_tokens, device=hidden_states.device)[None].expand(batch_size, -1))
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
                    layer_outputs = self._gradient_checkpointing_func(encoder_layer.__call__, hidden_states, global_hidden_states, attention_mask, output_attentions)
                else:
                    layer_outputs = encoder_layer(hidden_states, global_hidden_states, attention_mask, output_attentions=output_attentions)
                hidden_states = layer_outputs[0]
                global_hidden_states = layer_outputs[1]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)
        hidden_states = hidden_states[:, :seq_len]
        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + ((hidden_states, global_hidden_states),)
        if not return_dict:
            return tuple((v for v in [hidden_states, encoder_states, all_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)