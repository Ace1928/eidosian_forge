import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_mega import MegaConfig
class MegaBlock(nn.Module):

    def __init__(self, config: MegaConfig):
        super().__init__()
        self.seq_len_dim = 1
        self.mega_layer = MegaMovingAverageGatedAttention(config)
        self.nffn = MegaNormalizedFeedForwardNetwork(config) if config.use_normalized_ffn else None
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f'{self} should be used as a decoder model if cross attention is added')
            self.cross_attn = MegaGatedCrossAttention(config)
        else:
            self.cross_attn = None

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.LongTensor]=None, causal_mask: Optional[torch.LongTensor]=None, encoder_hidden_states: Optional[torch.FloatTensor]=None, encoder_attention_mask: Optional[torch.FloatTensor]=None, past_key_value: Optional[Tuple[torch.FloatTensor]]=None, output_attentions: Optional[bool]=False, use_cache: bool=False) -> Tuple[torch.Tensor]:
        """
        A single Mega layer: either encoder or decoder, with optional cross-attention and optional normalized
        feed-forward layer

        Args:
            hidden_states (`torch.Tensor` of shape `(target_sequence_length, batch_size, hidden_size)`):
                Hidden states to be updated by the Mega block
            attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
                Indicates which entries in the self/target sequence are to be ignored (mostly due to padding), where
                elements are either 1 for *not masked* or 0 for *masked*. Causal attention is enforced internally.
            causal_mask (`torch.LongTensor` of shape `(sequence_length, sequence_length)`, *optional*):
                Indicates which inputs are to be ignored due to causal attention, where elements are either 1 for *not
                masked* or 0 for *masked*
            encoder_hidden_states (`torch.Tensor`, of shape `(source_sequence_length, batch_size, hidden_size)`, *optional*):
                Encoder hidden states to be used for cross-attention (and required for encoder-decoder model setup)
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, source_sequence_length)`, *optional*):
                Indicates which entries in the cross/source sequence are to be ignored (mostly due to padding), where
                elements are either 1 for *not masked* or 0 for *masked*.
            past_key_value (`tuple(torch.Tensor)`, *optional*):
                The hidden states returned from the previous timestep during incremental decoding; expects that
                self-attention key, value, and EMA states are the first 3 entries in the tuple, and (if doing
                cross-attention) cross-attention key and value are the last 2 entries in the tuple
            output_attentions (`bool`, default `False`):
                Whether to return self-attention weights
            use_cache (`bool`, default `False`):
                Whether to perfom incremental decoding; uses `past_key_value` as prior state, and returns the updated
                states for use in the next step

        Returns:
            `tuple(torch.FloatTensor)` containing various elements depending on configuration ([`MegaConfig`]) and
            inputs:
            - **hidden_states** (`torch.FloatTensor` of shape `(target_sequence_length, batch_size, hidden_size)`) --
              Hidden states from target sequence updated by Mega
            - **self_attn_weights** (*optional*, returned when `output_attentions=True`) `torch.FloatTensor` of shape
              `(batch_size, 1, target_sequence_length, target_sequence_length)` -- The self-attention weights
              corresponding to how each token in the input sequence attends to every other token
            - **cross_attn_weights** (*optional*, returned when `output_attentions=True` and
              `config.add_cross_attention=True`) `torch.FloatTensor` of shape `(batch_size, source_sequence_length,
              target_sequence_length)` -- Pairwise cross-attention weights between every entry in the source sequence
              and target sequence
            - **self_key** (*optional*, returned when `use_cache=True`) `torch.FloatTensor` of shape `(batch_size,
              sequence_length, config.shared_representation_size)` -- The self-attention key state for use in the next
              step of incremental decoding
            - **self_value** (*optional*, returned when `use_cache=True`) `torch.FloatTensor` of shape `(batch_size,
              sequence_length, config.hidden_size)` -- The self-attention value state for use in the next step of
              incremental decoding
            - **self_ema_state** (*optional*, returned when `use_cache=True`) `torch.FloatTensor` of shape
              `(batch_size, config.ndim)` The incremental EMA state for use in the next step of incremental decoding.
            - **cross_key** (*optional*, returned when `use_cache=True` and `config.is_decoder=True`)
              `torch.FloatTensor` of shape `(batch_size, source_sequence_length, config.shared_representation_size)` --
              The cross-attention key state for use in the next step of incremental decoding
            - **cross_value** (*optional*, returned when `use_cache=True` and `config.is_decoder=True`)
              `torch.FloatTensor` of shape `(batch_size, source_sequence_length, config.hidden_size)` -- The
              cross-attention value state for use in the next step of incremental decoding
        """
        if use_cache and past_key_value is not None and (attention_mask is not None):
            mega_padding_mask = attention_mask[:, -1].unsqueeze(-1)
        else:
            mega_padding_mask = attention_mask
        mega_outputs = self.mega_layer(input=hidden_states, padding_mask=mega_padding_mask, causal_mask=causal_mask, past_key_values=past_key_value, output_attentions=output_attentions, use_cache=use_cache)
        new_hidden_states = mega_outputs[0]
        self_key, self_value, self_ema_state = mega_outputs[-3:] if use_cache else (None, None, None)
        self_attention_weights = mega_outputs[1] if output_attentions else None
        if self.cross_attn is not None:
            if encoder_hidden_states is None:
                raise ValueError('Requested cross-attention without providing encoder hidden states')
            cross_attn_outputs = self.cross_attn(query=new_hidden_states, key=encoder_hidden_states, value=encoder_hidden_states, key_padding_mask=encoder_attention_mask, past_key_values=past_key_value, output_attentions=output_attentions, use_cache=use_cache)
            new_hidden_states = cross_attn_outputs[0]
            cross_key, cross_value = cross_attn_outputs[-2:] if use_cache else (None, None)
            cross_attention_weights = cross_attn_outputs[1] if output_attentions else None
        if self.nffn is not None:
            new_hidden_states = self.nffn(new_hidden_states)
        outs = (new_hidden_states,)
        if output_attentions:
            outs = outs + (self_attention_weights,)
            if self.cross_attn is not None:
                outs = outs + (cross_attention_weights,)
        if use_cache:
            new_key_values = (self_key, self_value, self_ema_state)
            if self.cross_attn is not None:
                new_key_values = new_key_values + (cross_key, cross_value)
            outs = outs + (new_key_values,)
        return outs