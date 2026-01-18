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
class MegaGatedCrossAttention(nn.Module):
    """
    Gated Structured State Attention for use in encoder-decoder model. See Mega paper for more details. Only
    modifications from original implementation are variable names, removing the unnecessary `before_attn_fn` and
    `static_kv` arguments, and the stateful representation of incremental decoder state.
    """

    def __init__(self, config: MegaConfig):
        super().__init__()
        self.config = config
        self.activation = ACT2FN[self.config.activation]
        self.attention_activation = self.config.attention_activation
        self.scaling = self.config.shared_representation_size ** (-0.5) if self.attention_activation == 'softmax' else None
        self.dropout = MegaDropout(self.config.dropout_prob, is_featurewise=self.config.use_feature_dropout)
        self.hidden_dropout = MegaDropout(self.config.hidden_dropout_prob, is_featurewise=self.config.use_feature_dropout)
        self.attention_dropout = MegaDropout(self.config.attention_probs_dropout_prob, is_featurewise=False)
        self.prenorm = self.config.normalize_before_mega
        self.norm = MegaSequenceNorm(self.config.normalization_type, self.config.hidden_size, affine=self.config.norm_affine)
        self.k_proj = nn.Linear(self.config.hidden_size, self.config.shared_representation_size)
        self.v_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.q_proj = nn.Linear(self.config.hidden_size, 2 * self.config.hidden_size + self.config.shared_representation_size)
        self.h_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        if self.config.relative_positional_bias == 'simple':
            self.rel_pos_bias = MegaSimpleRelativePositionalBias(config)
        elif self.config.relative_positional_bias == 'rotary':
            self.rel_pos_bias = MegaRotaryRelativePositionalBias(config)
        else:
            raise ValueError('unknown relative position bias: {}'.format(self.config.relative_positional_bias))
        self.softmax = nn.Softmax(dim=-1)

    def element_attention(self, query, key, key_padding_mask, pidx):
        bsz, src_len, _ = key.size()
        tgt_len = query.size(1) if pidx is None else pidx + 1
        if key_padding_mask is not None:
            lengths = key_padding_mask.sum(dim=-1).view(bsz, 1, 1)
        else:
            lengths = src_len
        bias = self.rel_pos_bias(max(tgt_len, src_len))[:, :src_len]
        if pidx is not None:
            if query.size(1) != 1:
                raise ValueError('Position offset provided with queries longer than 1 token')
            bias = bias[pidx]
        else:
            bias = bias[:tgt_len]
        qk = torch.bmm(query, key.transpose(1, 2)) / lengths + bias
        attn_weights = ACT2FN[self.attention_activation](qk).type_as(qk)
        if key_padding_mask is not None:
            attn_weights = attn_weights * key_padding_mask.unsqueeze(1)
        return attn_weights

    def softmax_attention(self, query, key, key_padding_mask, pidx):
        bsz, src_len, _ = key.size()
        tgt_len = query.size(1) if pidx is None else pidx + 1
        bias = self.rel_pos_bias(max(tgt_len, src_len))[:, :src_len]
        if pidx is not None:
            if query.size(1) != 1:
                raise ValueError('Position offset provided with queries longer than 1 token')
            bias = bias[pidx]
        else:
            bias = bias[:tgt_len]
        query = query * self.scaling
        qk = torch.bmm(query, key.transpose(1, 2)) + bias
        if key_padding_mask is not None:
            qk = qk.masked_fill((1 - key_padding_mask).unsqueeze(1).to(torch.bool), float('-inf'))
        attn_weights = self.softmax(qk).type_as(qk)
        return attn_weights

    def forward(self, query, key: Optional[torch.Tensor], value: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor]=None, past_key_values: Optional[Tuple[torch.Tensor]]=None, output_attentions: bool=False, use_cache: bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Gated cross-attention used in Mega

        Args:
            query (`torch.Tensor` of shape `(target_sequence_length, batch_size, hidden_size)`):
                The self (or target) sequence input used as query inputs for cross-attention
            key (`torch.Tensor` of shape `(source_sequence_length, batch_size, hidden_size)`):
                The cross (or source) sequence input with shape used as keys in cross-attention
            value (`torch.Tensor` of shape `(source_sequence_length, batch_size, hidden_size)`):
                The cross (or source) sequence input with shape used as values in cross-attention
            key_padding_mask (`torch.LongTensor` of shape `(batch_size, source_sequence_length)`, *optional*):
                Padding mask corresponding to the source sequence, where entries are 1 for *not masked* and 0 for
                *masked* tokens
            past_key_values (`tuple(torch.FloatTensor)`, *optional*):
                If provided, the hidden state returned from the previous timestep during incremental decoding; expects
                that prior cross-attention keys and values will be the last two items in the tuple
            output_attentions (`bool`, defaults to `False`):
                Whether or not to return the cross-attention weights.
            use_cache (`bool`, defaults to `False`):
                Whether to perfom incremental decoding; uses `prev_state` as the prior timestep, and returns the
                updated EMA hidden state for use in the next step

        Returns:
            `tuple(torch.FloatTensor)` containing various elements depending on configuration ([`MegaConfig`]) and
            inputs:
            - **hidden_states** (`torch.FloatTensor` of shape `(target_sequence_length, batch_size, hidden_size)`) --
              Hidden states from target sequence updated by gated cross-attention
            - **attn_weights** (*optional*, returned when `output_attentions=True`) `torch.FloatTensor` of shape
              `(batch_size, source_sequence_length, target_sequence_length)` -- The pairwise cross-attention weights
              corresponding to each token in the source and target sequences
            - **cross_key** (*optional*, returned when `use_cache=True`) `torch.FloatTensor` of shape `(batch_size,
              source_sequence_length, config.shared_representation_size)` -- The cross-attention key state for use in
              the next step of incremental decoding
            - **cross_value** (*optional*, returned when `use_cache=True`) `torch.FloatTensor` of shape `(batch_size,
              source_sequence_length, config.hidden_size)` -- The cross-attention value state for use in the next step
              of incremental decoding
        """
        seq_len, bsz, embed_dim = query.size()
        if embed_dim != self.config.hidden_size:
            raise ValueError(f'Unexpected embedding dimension received: input is {embed_dim} but expected {self.config.hidden_size}')
        if past_key_values is not None:
            if seq_len != 1:
                raise ValueError(f'Incremental decoding requested with self-sequence length > 1: {seq_len}')
            prev_cross_key, prev_cross_value = past_key_values[-2:]
            key = value = None
            prev_self_key = past_key_values[0]
            num_incremental_steps = prev_self_key.size(1) + 1
        else:
            prev_cross_key = prev_cross_value = None
            num_incremental_steps = 0 if use_cache and seq_len == 1 else None
        full_query = query
        if self.prenorm:
            full_query = self.norm(full_query)
        query_projected = self.q_proj(full_query)
        residual_weight, target_gate, attention_query = torch.split(query_projected, [self.config.hidden_size, self.config.hidden_size, self.config.shared_representation_size], dim=-1)
        residual_weight = torch.sigmoid(residual_weight)
        target_gate = F.silu(target_gate)
        if key is None:
            if value is not None:
                raise ValueError('Key and value must be `None` simultaneously')
            projected_key = projected_value = None
        else:
            projected_key = self.k_proj(key)
            projected_value = self.activation(self.v_proj(key))
        attention_query = attention_query.transpose(0, 1)
        if projected_key is not None:
            projected_key = projected_key.transpose(0, 1)
        if projected_value is not None:
            projected_value = projected_value.transpose(0, 1)
        if past_key_values is not None:
            projected_key = prev_cross_key
            projected_value = prev_cross_value
        if use_cache:
            updated_cross_key = projected_key
            updated_cross_value = projected_value
        ctx_len = projected_key.size(1)
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        if key_padding_mask is not None:
            if key_padding_mask.size(0) != bsz:
                raise ValueError('Key padding mask does not align on the batch dimension')
            if key_padding_mask.size(1) != ctx_len:
                raise ValueError('Key padding mask does not align on the sequence length dimension')
        if self.attention_activation == 'softmax':
            attn_weights = self.softmax_attention(attention_query, projected_key, key_padding_mask, num_incremental_steps)
        else:
            attn_weights = self.element_attention(attention_query, projected_key, key_padding_mask, num_incremental_steps)
        projected_value = self.hidden_dropout(projected_value, batch_first=True)
        kernel = self.attention_dropout(attn_weights)
        weighted_targets = torch.bmm(kernel, projected_value).transpose(0, 1)
        weighted_targets = self.activation(self.h_proj(weighted_targets * target_gate))
        weighted_targets = self.dropout(weighted_targets)
        out = torch.addcmul(query, residual_weight, weighted_targets - query)
        if not self.prenorm:
            out = self.norm(out)
        outputs = (out, attn_weights) if output_attentions else (out,)
        if use_cache:
            outputs = outputs + (updated_cross_key, updated_cross_value)
        return outputs