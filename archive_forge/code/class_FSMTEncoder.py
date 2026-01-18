import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss, LayerNorm
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_fsmt import FSMTConfig
class FSMTEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a [`EncoderLayer`].

    Args:
        config: FSMTConfig
    """

    def __init__(self, config: FSMTConfig, embed_tokens):
        super().__init__()
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        self.padding_idx = embed_tokens.padding_idx
        self.embed_tokens = embed_tokens
        embed_dim = embed_tokens.embedding_dim
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.embed_positions = SinusoidalPositionalEmbedding(config.max_position_embeddings + self.padding_idx + 1, embed_dim, self.padding_idx)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, inputs_embeds: torch.Tensor=None, head_mask: Optional[torch.Tensor]=None, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        """
        Args:
            input_ids (`torch.LongTensor`): tokens in the source language of shape
                *(batch, src_len)*
            attention_mask (`torch.LongTensor`): indicating which indices are padding tokens
            inputs_embeds (`torch.FloatTensor`):
                embedding vectors of shape *(batch, src_len, embed_dim)*
            head_mask (`torch.Tensor` of shape `(num_layers, num_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

        Returns:
            BaseModelOutput or Tuple comprised of:

                - **x** (`torch.Tensor`): the last encoder layer's output of shape *(src_len, batch, embed_dim)*
                - **encoder_states** (`Tuple(torch.FloatTensor`)): all intermediate hidden states of shape *(src_len,
                  batch, embed_dim)*. Only populated if *output_hidden_states:* is True.
                - **all_attentions** (`Tuple(torch.FloatTensor`)): Attention weights for each layer.
                During training might not be of length n_layers because of layer dropout.
        """
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
            embed_pos = self.embed_positions(input_ids)
        elif inputs_embeds is not None:
            inputs_embeds = inputs_embeds * self.embed_scale
            position_ids = inputs_embeds[:, :, 0].masked_fill(inputs_embeds[:, :, 0].eq(0), self.embed_positions.padding_idx)
            embed_pos = self.embed_positions(position_ids)
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        x = inputs_embeds + embed_pos
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(0, 1)
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        if head_mask is not None:
            assert head_mask.size()[0] == len(self.layers), f'The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}.'
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                x = x.transpose(0, 1)
                encoder_states += (x,)
                x = x.transpose(0, 1)
            dropout_probability = torch.rand([])
            if self.training and dropout_probability < self.layerdrop:
                attn = None
            else:
                x, attn = encoder_layer(x, attention_mask, layer_head_mask=head_mask[idx] if head_mask is not None else None, output_attentions=output_attentions)
            if output_attentions:
                all_attentions = all_attentions + (attn,)
        x = x.transpose(0, 1)
        if output_hidden_states:
            encoder_states += (x,)
        if not return_dict:
            return tuple((v for v in [x, encoder_states, all_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=x, hidden_states=encoder_states, attentions=all_attentions)