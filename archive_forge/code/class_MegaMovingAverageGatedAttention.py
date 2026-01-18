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
class MegaMovingAverageGatedAttention(nn.Module):
    """
    Pure PyTorch implementation of Mega block; see https://arxiv.org/abs/2209.10655 and original fairseq implementation
    at https://github.com/facebookresearch/mega (copyright Meta Research, licensed under MIT License)

    Differences from original implementation include hidden state refactor and fixed inconsistency with additive /
    multiplicative attention masks
    """

    def __init__(self, config: MegaConfig):
        super().__init__()
        self.config = config
        self.activation = ACT2FN[self.config.activation]
        self.scaling = self.config.shared_representation_size ** (-0.5) if self.config.attention_activation == 'softmax' else None
        self.dropout = MegaDropout(self.config.dropout_prob, is_featurewise=self.config.use_feature_dropout)
        self.hidden_dropout = MegaDropout(self.config.hidden_dropout_prob, is_featurewise=self.config.use_feature_dropout)
        self.attention_dropout = MegaDropout(self.config.attention_probs_dropout_prob, is_featurewise=False)
        self.norm = MegaSequenceNorm(self.config.normalization_type, self.config.hidden_size, affine=self.config.norm_affine)
        self.ema_gate = MegaMultiDimensionDampedEma(config)
        self.v_proj = nn.Linear(self.config.hidden_size, self.config.intermediate_size)
        self.mx_proj = nn.Linear(self.config.hidden_size, self.config.shared_representation_size + self.config.intermediate_size + 2 * self.config.hidden_size)
        self.h_proj = nn.Linear(self.config.intermediate_size, self.config.hidden_size)
        self.qk_weight = nn.Parameter(torch.Tensor(2, self.config.shared_representation_size))
        self.qk_bias = nn.Parameter(torch.Tensor(2, self.config.shared_representation_size))
        if self.config.relative_positional_bias == 'simple':
            self.rel_pos_bias = MegaSimpleRelativePositionalBias(config)
        elif self.config.relative_positional_bias == 'rotary':
            self.rel_pos_bias = MegaRotaryRelativePositionalBias(config)
        else:
            raise ValueError(f'Unknown relative positional bias: {self.config.relative_positional_bias}')
        self.softmax = nn.Softmax(dim=-1)
        self.attention_function = self.softmax_attention if self.config.attention_activation == 'softmax' else self.element_attention

    def element_attention(self, query, key, padding_mask, causal_mask):
        """
        Apply element-wise attention via relu^2 or laplace. Same as original implementation but with standardized
        causal attention mask. Expects the Hugging Face standard attention mask paradigm: 1 for not masked, and 0 for
        masked.
        """
        seq_len = key.size(2)
        if padding_mask is not None:
            lengths = padding_mask.sum(-1, keepdim=True)
            lengths = lengths.clamp(min=1.0).unsqueeze(-1)
        else:
            lengths = seq_len
        if causal_mask is not None:
            lengths = causal_mask.sum(dim=-1, keepdim=True)
        bias = self.rel_pos_bias(seq_len)
        if seq_len != query.size(2):
            if query.size(2) != 1:
                raise ValueError('Size mismatch between Q and K in element attention')
            bias = bias[-1:]
        qk = torch.matmul(query, key.transpose(2, 3)) / lengths + bias
        attn_weights = ACT2FN[self.config.attention_activation](qk).type_as(qk)
        if padding_mask is not None:
            attn_weights = attn_weights * padding_mask.unsqueeze(2)
        if causal_mask is not None:
            attn_weights = attn_weights * causal_mask
        return attn_weights

    def softmax_attention(self, query, key, padding_mask, causal_mask):
        """Standard softmax self-attention, as in the original Transformer paper"""
        seq_len = key.size(2)
        bias = self.rel_pos_bias(seq_len)
        if seq_len != query.size(2):
            if query.size(2) != 1:
                raise ValueError('Size mismatch between Q and K in softmax attention')
            bias = bias[-1:]
        query = query * self.scaling
        qk = torch.matmul(query, key.transpose(2, 3)) + bias
        if causal_mask is not None:
            additive_causal_mask = torch.zeros_like(causal_mask, dtype=qk.dtype)
            additive_causal_mask = additive_causal_mask.masked_fill((1 - causal_mask).bool(), float('-inf'))
            qk = qk + additive_causal_mask
        if padding_mask is not None:
            padding_mask = 1 - padding_mask
            padding_mask_all = padding_mask.all(dim=-1, keepdim=True)
            padding_mask = torch.logical_and(padding_mask, ~padding_mask_all)
            qk = qk.masked_fill(padding_mask.unsqueeze(2).to(torch.bool), float('-inf'))
        attn_weights = self.softmax(qk).type_as(qk)
        return attn_weights

    def forward(self, input, padding_mask: Optional[torch.Tensor]=None, causal_mask: Optional[torch.Tensor]=None, past_key_values: Optional[Tuple[torch.Tensor]]=None, output_attentions=False, use_cache=False):
        """
        Mega's self-attention block, which combines multi-headed EMA with traditional self-attention

        Args:
            input (`torch.Tensor` of shape `(sequence_length, batch_size, hidden_size)`):
                Hidden states to be updated by Mega's self-attention
            padding_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indicates which inputs are to be ignored due to padding, where elements are either 1 for *not masked*
                or 0 for *masked*
            causal_mask (`torch.LongTensor` of shape `(sequence_length, sequence_length)`, *optional*):
                Indicates which inputs are to be ignored due to causal attention, where elements are either 1 for *not
                masked* or 0 for *masked*
            past_key_values (`tuple(torch.Tensor)`, *optional*):
                The hidden states returned from the previous timestep during incremental decoding; expects that
                self-attention key, value, and EMA states are the first 3 entries in the tuple
            output_attentions (`bool`, default `False`):
                Whether to return self-attention weights
            use_cache (`bool`, default `False`):
                Whether to perfom incremental decoding; uses `past_key_values` as prior state, and returns the updated
                states for use in the next step

        Returns:
            `tuple(torch.FloatTensor)` containing various elements depending on configuration ([`MegaConfig`]) and
            inputs:
            - **hidden_states** (`torch.FloatTensor` of shape `(sequence_length, batch_size, hidden_size)`) -- Hidden
              states from target sequence updated by Mega's self-attention
            - **attn_weights** (*optional*, returned when `output_attentions=True`) `torch.FloatTensor` of shape
              `(batch_size, 1, sequence_length, sequence_length)` -- The self-attention weights corresponding to how
              each token in the input sequence attends to every other token
            - **self_key** (*optional*, returned when `use_cache=True`) `torch.FloatTensor` of shape `(batch_size,
              sequence_length, config.shared_representation_size)` -- The self-attention key state for use in the next
              step of incremental decoding
            - **self_value** (*optional*, returned when `use_cache=True`) `torch.FloatTensor` of shape `(batch_size,
              sequence_length, config.hidden_size)` -- The self-attention value state for use in the next step of
              incremental decoding
            - **self_ema_state** (*optional*, returned when `use_cache=True`) `torch.FloatTensor` of shape
              `(batch_size, config.ndim)` The incremental EMA state for use in the next step of incremental decoding.
        """
        seq_len, bsz, embed_dim = input.size()
        if embed_dim != self.config.hidden_size:
            raise ValueError(f'Input embedding dimension should be {self.config.hidden_size}; received {embed_dim}')
        residual = input
        if self.config.normalize_before_mega:
            input = self.norm(input)
        value = self.activation(self.v_proj(input))
        if self.config.is_decoder and past_key_values is not None:
            if seq_len > 1:
                raise ValueError(f'Incremental decoding only supports self sequence length of 1; received {seq_len}')
            prev_self_key, prev_self_value, prev_ema_state = past_key_values[0:3]
        else:
            prev_self_key = prev_self_value = prev_ema_state = None
        ema_out, updated_ema_state = self.ema_gate(input, attention_mask=padding_mask, prev_state=prev_ema_state, use_cache=use_cache)
        ema_out = self.dropout(ema_out)
        base = self.mx_proj(ema_out)
        residual_weight, query_key_gates, intermediate_state = torch.split(base, [self.config.hidden_size, self.config.shared_representation_size + self.config.intermediate_size, self.config.hidden_size], dim=-1)
        residual_weight = torch.sigmoid(residual_weight)
        query_key_gates = F.silu(query_key_gates)
        query_key, attention_gate = torch.split(query_key_gates, [self.config.shared_representation_size, self.config.intermediate_size], dim=-1)
        query_key = query_key.unsqueeze(2) * self.qk_weight + self.qk_bias
        query, key = torch.unbind(query_key, dim=2)
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        if self.config.is_decoder:
            if prev_self_key is not None:
                key = torch.cat([prev_self_key, key], dim=1)
            if prev_self_value is not None:
                value = torch.cat([prev_self_value, value], dim=1)
            if not self.config.use_chunking:
                updated_self_key = key
                updated_self_value = value
            else:
                curr_len = key.size(1) % self.config.chunk_size
                if curr_len == 0:
                    updated_self_key = None
                    updated_self_value = None
                else:
                    updated_self_key = key
                    updated_self_value = value
        ctx_len = key.size(1)
        if not self.config.use_chunking:
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
            if padding_mask is not None:
                padding_mask = padding_mask.unsqueeze(1)
        else:
            if seq_len < self.config.chunk_size:
                query = query.unsqueeze(1)
            else:
                n_chunks = seq_len // self.config.chunk_size
                query = query.reshape(bsz, n_chunks, self.config.chunk_size, self.config.shared_representation_size)
            if ctx_len < self.config.chunk_size:
                key = key.unsqueeze(1)
                value = value.unsqueeze(1)
                if padding_mask is not None:
                    padding_mask = padding_mask.unsqueeze(1)
            else:
                n_chunks = ctx_len // self.config.chunk_size
                key = key.reshape(bsz, n_chunks, self.config.chunk_size, self.config.shared_representation_size)
                value = value.reshape(bsz, n_chunks, self.config.chunk_size, self.config.intermediate_size)
                if padding_mask is not None:
                    padding_mask = padding_mask.view(bsz, n_chunks, self.config.chunk_size)
        if padding_mask is not None and padding_mask.dim() == 0:
            padding_mask = None
        attn_weights = self.attention_function(query, key, padding_mask=padding_mask, causal_mask=causal_mask)
        value = self.hidden_dropout(value, batch_first=True)
        kernel = self.attention_dropout(attn_weights)
        weighted_self_output = torch.matmul(kernel, value).view(bsz, seq_len, self.config.intermediate_size).transpose(0, 1)
        weighted_self_output = self.activation(intermediate_state + self.h_proj(weighted_self_output * attention_gate))
        weighted_self_output = self.dropout(weighted_self_output)
        out = torch.addcmul(residual, residual_weight, weighted_self_output - residual)
        if not self.config.normalize_before_mega:
            out = self.norm(out)
        return_values = (out, attn_weights) if output_attentions else (out,)
        if self.config.is_decoder:
            return_values = return_values + (updated_self_key, updated_self_value, updated_ema_state)
        return return_values