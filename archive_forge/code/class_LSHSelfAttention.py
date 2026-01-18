import sys
from collections import namedtuple
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.autograd.function import Function
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import CausalLMOutput, MaskedLMOutput, QuestionAnsweringModelOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_reformer import ReformerConfig
class LSHSelfAttention(nn.Module, EfficientAttentionMixin):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.chunk_length = config.lsh_attn_chunk_length
        self.num_hashes = config.num_hashes
        self.num_buckets = config.num_buckets
        self.num_chunks_before = config.lsh_num_chunks_before
        self.num_chunks_after = config.lsh_num_chunks_after
        self.hash_seed = config.hash_seed
        self.is_decoder = config.is_decoder
        self.max_position_embeddings = config.max_position_embeddings
        self.dropout = config.lsh_attention_probs_dropout_prob
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.hidden_size = config.hidden_size
        self.query_key = nn.Linear(self.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(self.hidden_size, self.all_head_size, bias=False)
        self.register_buffer('self_mask_value_float16', torch.tensor(-1000.0), persistent=False)
        self.register_buffer('self_mask_value_float32', torch.tensor(-100000.0), persistent=False)
        self.register_buffer('mask_value_float16', torch.tensor(-10000.0), persistent=False)
        self.register_buffer('mask_value_float32', torch.tensor(-1000000000.0), persistent=False)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, num_hashes=None, buckets=None, past_buckets_states=None, use_cache=False, output_attentions=False, **kwargs):
        sequence_length = hidden_states.shape[1]
        batch_size = hidden_states.shape[0]
        num_hashes = num_hashes if num_hashes is not None else self.num_hashes
        do_cached_attention = use_cache and past_buckets_states[1] is not None
        if do_cached_attention:
            assert sequence_length == 1, f'At the moment, auto-regressive language generation is only possible one word at a time. Make sure that input sequence length {sequence_length} equals 1, when `past_buckets_states` is passed.'
            past_buckets = past_buckets_states[0]
            past_states = past_buckets_states[1]
            query_vectors = self.query_key(hidden_states)
            query_vectors = self._split_hidden_size_dim(query_vectors, self.num_attention_heads, self.attention_head_size)
            if past_buckets is not None:
                key_value_hidden_states, sorted_bucket_idx, buckets = self._get_relevant_hid_states_and_buckets(query_vectors=query_vectors, attention_mask=attention_mask, num_hashes=num_hashes, hidden_states=hidden_states, past_states=past_states, past_buckets=past_buckets)
                query_key_vectors = self._query_per_attn_head(key_value_hidden_states)
                value_vectors = self._value_per_attn_head(key_value_hidden_states)
                query_key_vectors = self._split_seq_length_dim_to(query_key_vectors, num_hashes, -1, self.num_attention_heads, self.attention_head_size)
                value_vectors = self._split_seq_length_dim_to(value_vectors, num_hashes, -1, self.num_attention_heads, self.attention_head_size)
                query_vectors = query_vectors.unsqueeze(2).repeat(1, 1, num_hashes, 1, 1)
            else:
                key_value_hidden_states = torch.cat([past_states, hidden_states], dim=1)
                query_key_vectors = self.query_key(key_value_hidden_states)
                value_vectors = self.value(key_value_hidden_states)
        else:
            query_vectors = None
            query_key_vectors = self.query_key(hidden_states)
            value_vectors = self.value(hidden_states)
        if not do_cached_attention or past_buckets is None:
            query_key_vectors = self._split_hidden_size_dim(query_key_vectors, self.num_attention_heads, self.attention_head_size)
            value_vectors = self._split_hidden_size_dim(value_vectors, self.num_attention_heads, self.attention_head_size)
        if do_cached_attention and past_buckets is None and (key_value_hidden_states.shape[1] >= self.chunk_length):
            buckets = self._hash_vectors(query_key_vectors, num_hashes, attention_mask)
        del hidden_states
        assert query_key_vectors.shape[-1] == self.attention_head_size, f'last dim of query_key_vectors is {query_key_vectors.shape[-1]} but should be {self.attention_head_size}.'
        assert value_vectors.shape[-1] == self.attention_head_size, f'last dim of value_vectors is {value_vectors.shape[-1]} but should be {self.attention_head_size}.'
        do_standard_self_attention = sequence_length <= self.chunk_length or (use_cache and past_buckets_states[1] is not None)
        if not do_standard_self_attention:
            if self.num_buckets is None:
                self._set_num_buckets(sequence_length)
            if buckets is None:
                buckets = self._hash_vectors(query_key_vectors, num_hashes, attention_mask)
            else:
                buckets = buckets.view(batch_size, self.num_attention_heads, num_hashes * sequence_length)
            assert int(buckets.shape[-1]) == num_hashes * sequence_length, f'last dim of buckets is {buckets.shape[-1]}, but should be {num_hashes * sequence_length}'
            sorted_bucket_idx, undo_sorted_bucket_idx = self._get_sorted_bucket_idx_and_undo_sorted_bucket_idx(sequence_length, buckets, num_hashes)
            sorted_bucket_idx_per_hash = sorted_bucket_idx % sequence_length
            query_key_vectors = self._gather_by_expansion(query_key_vectors, sorted_bucket_idx_per_hash, num_hashes)
            value_vectors = self._gather_by_expansion(value_vectors, sorted_bucket_idx_per_hash, num_hashes)
            query_key_vectors = self._split_seq_length_dim_to(query_key_vectors, -1, self.chunk_length, self.num_attention_heads, self.attention_head_size)
            value_vectors = self._split_seq_length_dim_to(value_vectors, -1, self.chunk_length, self.num_attention_heads, self.attention_head_size)
            if self.chunk_length is None:
                assert self.num_chunks_before == 0 and self.num_chunks_after == 0, 'If `config.chunk_length` is `None`, make sure `config.num_chunks_after` and `config.num_chunks_before` are set to 0.'
        elif do_cached_attention and past_buckets is not None:
            sorted_bucket_idx_per_hash = sorted_bucket_idx
        else:
            sorted_bucket_idx_per_hash = torch.arange(sequence_length, device=query_key_vectors.device).repeat(batch_size, self.num_attention_heads, 1)
        sqrt_num = np.sqrt(self.attention_head_size)
        key_vectors = self._len_and_dim_norm(query_key_vectors, sqrt_num)
        query_vectors = query_vectors if query_vectors is not None else query_key_vectors
        del query_key_vectors
        out_vectors, logits, attention_probs = self._attend(query_vectors=query_vectors, key_vectors=key_vectors, value_vectors=value_vectors, sorted_bucket_idx_per_hash=sorted_bucket_idx_per_hash, attention_mask=attention_mask, head_mask=head_mask, do_standard_self_attention=do_standard_self_attention, do_cached_attention=do_cached_attention)
        del key_vectors, value_vectors
        if not do_standard_self_attention:
            out_vectors, logits = ReverseSort.apply(out_vectors, logits, sorted_bucket_idx, undo_sorted_bucket_idx)
        if not do_standard_self_attention or (do_cached_attention and past_buckets is not None):
            if num_hashes > 1:
                out_vectors = self._split_seq_length_dim_to(out_vectors, num_hashes, sequence_length, self.num_attention_heads, self.attention_head_size)
                logits = self._split_seq_length_dim_to(logits, num_hashes, sequence_length, self.num_attention_heads, self.attention_head_size).unsqueeze(-1)
                probs_vectors = torch.exp(logits - torch.logsumexp(logits, dim=2, keepdim=True))
                out_vectors = torch.sum(out_vectors * probs_vectors, dim=2)
                del probs_vectors
            del logits
        assert out_vectors.shape == (batch_size, self.num_attention_heads, sequence_length, self.attention_head_size), 'out_vectors have be of shape `[batch_size, config.num_attention_heads, sequence_length, config.attention_head_size]`.'
        out_vectors = self._merge_hidden_size_dims(out_vectors, self.num_attention_heads, self.attention_head_size)
        if output_attentions is False:
            attention_probs = ()
        if buckets is not None:
            buckets = buckets.view(batch_size, self.num_attention_heads, num_hashes, -1)
        return LSHSelfAttentionOutput(hidden_states=out_vectors, attention_probs=attention_probs, buckets=buckets)

    def _query_per_attn_head(self, hidden_states):
        per_head_query_key = self.query_key.weight.reshape(self.num_attention_heads, self.attention_head_size, self.hidden_size).transpose(-2, -1)
        query_key_vectors = torch.einsum('balh,ahr->balr', hidden_states, per_head_query_key)
        return query_key_vectors

    def _value_per_attn_head(self, hidden_states):
        per_head_value = self.value.weight.reshape(self.num_attention_heads, self.attention_head_size, self.hidden_size).transpose(-2, -1)
        value_vectors = torch.einsum('balh,ahr->balr', hidden_states, per_head_value)
        return value_vectors

    def _hash_vectors(self, vectors, num_hashes, attention_mask, increase_num_buckets=False):
        batch_size = vectors.shape[0]
        if isinstance(self.num_buckets, int):
            assert self.num_buckets % 2 == 0, f'There should be an even number of buckets, but `self.num_buckets`: {self.num_buckets}'
            rotation_size = self.num_buckets
            num_buckets = self.num_buckets
        else:
            rotation_size, num_buckets = (0, 1)
            for bucket_factor in self.num_buckets:
                assert bucket_factor % 2 == 0, f'The number of buckets should be even, but `num_bucket`: {bucket_factor}'
                rotation_size = rotation_size + bucket_factor
                num_buckets = num_buckets * bucket_factor
        vectors = vectors.detach()
        if self.hash_seed is not None:
            torch.manual_seed(self.hash_seed)
        rotations_shape = (self.num_attention_heads, vectors.shape[-1], num_hashes, rotation_size // 2)
        random_rotations = torch.randn(rotations_shape, device=vectors.device, dtype=vectors.dtype)
        rotated_vectors = torch.einsum('bmtd,mdhr->bmhtr', vectors, random_rotations)
        if isinstance(self.num_buckets, int) or len(self.num_buckets) == 1:
            rotated_vectors = torch.cat([rotated_vectors, -rotated_vectors], dim=-1)
            buckets = torch.argmax(rotated_vectors, dim=-1)
        else:
            buckets, cur_sum, cur_product = (None, 0, 1)
            for bucket_factor in self.num_buckets:
                rotated_vectors_factor = rotated_vectors[..., cur_sum:cur_sum + bucket_factor // 2]
                cur_sum = cur_sum + bucket_factor // 2
                rotated_vectors_factor = torch.cat([rotated_vectors_factor, -rotated_vectors_factor], dim=-1)
                if buckets is None:
                    buckets = torch.argmax(rotated_vectors_factor, dim=-1)
                else:
                    buckets = buckets + cur_product * torch.argmax(rotated_vectors_factor, dim=-1)
                cur_product = cur_product * bucket_factor
        if attention_mask is not None and attention_mask.sum().item() < batch_size * attention_mask.shape[-1]:
            num_buckets = num_buckets + 1
            buckets_mask = attention_mask.to(torch.bool)[:, None, None, :].expand(buckets.shape)
            buckets = torch.where(buckets_mask, buckets, torch.tensor(num_buckets - 1, dtype=torch.long, device=buckets.device))
        elif increase_num_buckets:
            num_buckets = num_buckets + 1
        offsets = torch.arange(num_hashes, device=vectors.device)
        offsets = (offsets * num_buckets).view((1, 1, -1, 1))
        offsets = offsets.expand((batch_size, self.num_attention_heads) + offsets.shape[-2:])
        offset_buckets = (buckets + offsets).flatten(start_dim=2, end_dim=3)
        return offset_buckets

    def _get_sorted_bucket_idx_and_undo_sorted_bucket_idx(self, sequence_length, buckets, num_hashes):
        with torch.no_grad():
            sorted_bucket_idx = _stable_argsort(buckets, dim=-1)
            indices = torch.arange(sorted_bucket_idx.shape[-1], device=buckets.device).view(1, 1, -1).expand(sorted_bucket_idx.shape)
            undo_sorted_bucket_idx = sorted_bucket_idx.new(*sorted_bucket_idx.size())
            undo_sorted_bucket_idx.scatter_(-1, sorted_bucket_idx, indices)
        return (sorted_bucket_idx, undo_sorted_bucket_idx)

    def _set_num_buckets(self, sequence_length):
        num_buckets_pow_2 = (2 * (sequence_length // self.chunk_length)).bit_length() - 1
        num_buckets = 2 ** num_buckets_pow_2
        num_buckets_limit = 2 * max(int((self.max_position_embeddings // self.chunk_length) ** 0.5), self.chunk_length)
        if num_buckets > num_buckets_limit:
            num_buckets = [2 ** (num_buckets_pow_2 // 2), 2 ** (num_buckets_pow_2 - num_buckets_pow_2 // 2)]
        logger.warning(f'config.num_buckets is not set. Setting config.num_buckets to {num_buckets}...')
        self.config.num_buckets = num_buckets
        self.num_buckets = num_buckets

    def _attend(self, query_vectors, key_vectors, value_vectors, sorted_bucket_idx_per_hash, attention_mask, head_mask, do_standard_self_attention, do_cached_attention):
        if not do_standard_self_attention:
            key_vectors = self._look_adjacent(key_vectors, self.num_chunks_before, self.num_chunks_after)
            value_vectors = self._look_adjacent(value_vectors, self.num_chunks_before, self.num_chunks_after)
        query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))
        del query_vectors, key_vectors
        if not do_standard_self_attention:
            query_bucket_idx = self._split_seq_length_dim_to(sorted_bucket_idx_per_hash, -1, self.chunk_length, self.num_attention_heads)
            key_value_bucket_idx = self._look_adjacent(query_bucket_idx, self.num_chunks_before, self.num_chunks_after)
        elif do_cached_attention and query_key_dots.ndim > 4:
            key_value_bucket_idx = sorted_bucket_idx_per_hash
            query_bucket_idx = key_value_bucket_idx.new_ones(key_value_bucket_idx.shape[:-1] + (1,)) * key_value_bucket_idx.max()
        elif do_cached_attention and query_key_dots.ndim <= 4:
            query_bucket_idx = (query_key_dots.shape[-1] - 1) * torch.ones_like(query_key_dots)[:, :, :, -1]
            key_value_bucket_idx = torch.arange(query_key_dots.shape[-1], dtype=torch.long, device=query_key_dots.device)[None, None, :].expand(query_bucket_idx.shape[:2] + (-1,))
        else:
            query_bucket_idx = key_value_bucket_idx = sorted_bucket_idx_per_hash
        if query_key_dots.dtype == torch.float16:
            self_mask_value = self.self_mask_value_float16.half()
            mask_value = self.mask_value_float16.half()
        else:
            self_mask_value = self.self_mask_value_float32
            mask_value = self.mask_value_float32
        if not do_cached_attention:
            mask = self._compute_attn_mask(query_bucket_idx, key_value_bucket_idx, attention_mask, query_key_dots.shape, do_standard_self_attention)
            if mask is not None:
                query_key_dots = torch.where(mask, query_key_dots, mask_value)
            del mask
        self_mask = torch.ne(query_bucket_idx.unsqueeze(-1), key_value_bucket_idx.unsqueeze(-2)).to(query_bucket_idx.device)
        query_key_dots = torch.where(self_mask, query_key_dots, self_mask_value)
        del self_mask
        logits = torch.logsumexp(query_key_dots, dim=-1, keepdim=True)
        attention_probs = torch.exp(query_key_dots - logits)
        del query_key_dots
        attention_probs = nn.functional.dropout(attention_probs, p=self.dropout, training=self.training)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        out_vectors = torch.matmul(attention_probs, value_vectors)
        del value_vectors
        if out_vectors.ndim > 4:
            logits = logits.flatten(start_dim=2, end_dim=3).squeeze(-1)
            out_vectors = out_vectors.flatten(start_dim=2, end_dim=3)
        return (out_vectors, logits, attention_probs)

    def _compute_attn_mask(self, query_indices, key_indices, attention_mask, query_key_dot_shape, do_standard_self_attention):
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.bool)[:, None, :]
            if not do_standard_self_attention:
                attention_mask = attention_mask[:, None, :]
                attention_mask = attention_mask.expand(query_indices.shape[:-1] + (-1,))
                attention_mask = torch.gather(attention_mask, -1, key_indices)
            attention_mask = attention_mask.unsqueeze(-2).expand(query_key_dot_shape)
        if self.is_decoder is True:
            causal_mask = torch.ge(query_indices.unsqueeze(-1), key_indices.unsqueeze(-2)).to(query_indices.device)
            if attention_mask is not None:
                attention_mask = causal_mask * attention_mask
            else:
                attention_mask = causal_mask
        return attention_mask

    def _get_relevant_hid_states_and_buckets(self, query_vectors, attention_mask, num_hashes, hidden_states, past_states, past_buckets):
        hidden_states = torch.cat([past_states, hidden_states], dim=1)
        batch_size = hidden_states.shape[0]
        sequence_length = hidden_states.shape[1]
        max_bucket = self.num_buckets if isinstance(self.num_buckets, int) else reduce(mul, self.num_buckets)
        increase_num_buckets = past_buckets.max() > num_hashes * max_bucket - 1
        query_buckets = self._hash_vectors(query_vectors, num_hashes, attention_mask, increase_num_buckets=increase_num_buckets)
        concat_buckets = torch.cat([past_buckets, query_buckets.unsqueeze(-1)], dim=-1)
        bucket_idx = _stable_argsort(concat_buckets, dim=-1)
        assert bucket_idx.shape == (batch_size, self.num_attention_heads, num_hashes, sequence_length), f'bucket_idx should have shape {(batch_size, self.num_attention_heads, num_hashes, sequence_length)}, but has shape {bucket_idx.shape}.'
        relevant_bucket_idx = (bucket_idx == bucket_idx.shape[-1] - 1).nonzero()
        relevant_bucket_idx_chunk = self._expand_to_indices_in_relevant_chunk(relevant_bucket_idx, sequence_length)
        relevant_bucket_idx_chunk = bucket_idx[tuple(relevant_bucket_idx_chunk.transpose(0, 1))]
        offset = torch.arange(relevant_bucket_idx_chunk.shape[-1], device=hidden_states.device, dtype=torch.long)
        bucket_idx_batch_offset = sequence_length * (batch_size * torch.div(offset, relevant_bucket_idx_chunk.shape[-1], rounding_mode='floor'))
        relevant_bucket_idx_chunk_all_batch = relevant_bucket_idx_chunk + bucket_idx_batch_offset
        hidden_states = hidden_states.reshape((-1, self.hidden_size))
        relevant_hidden_states = hidden_states.index_select(0, relevant_bucket_idx_chunk_all_batch)
        relevant_hidden_states = relevant_hidden_states.reshape(batch_size, self.num_attention_heads, -1, self.hidden_size)
        relevant_bucket_idx_chunk = relevant_bucket_idx_chunk.reshape(batch_size, self.num_attention_heads, num_hashes, -1)
        assert relevant_hidden_states.shape[2] == (self.num_chunks_before + self.num_chunks_after + 1) * self.chunk_length * num_hashes, f'There should be {(self.num_chunks_before + self.num_chunks_after + 1) * self.chunk_length * num_hashes} `hidden_states`, there are {relevant_hidden_states.shape[2]} `hidden_states`.'
        assert relevant_bucket_idx_chunk.shape[-1] == (self.num_chunks_before + self.num_chunks_after + 1) * self.chunk_length, f'There should be {(self.num_chunks_before + self.num_chunks_after + 1) * self.chunk_length} `hidden_states`, there are {relevant_bucket_idx_chunk.shape[-1]} `bucket_idx`.'
        return (relevant_hidden_states, relevant_bucket_idx_chunk, query_buckets)

    def _expand_to_indices_in_relevant_chunk(self, indices, sequence_length):
        start_indices_chunk = (indices[:, -1] // self.chunk_length - self.num_chunks_before) * self.chunk_length
        total_chunk_size = self.chunk_length * (1 + self.num_chunks_before + self.num_chunks_after)
        expanded_start_indices = start_indices_chunk.unsqueeze(-1).expand(indices.shape[0], total_chunk_size)
        chunk_sequence_indices = expanded_start_indices + torch.arange(total_chunk_size, device=indices.device, dtype=torch.long).unsqueeze(0).expand(indices.shape[0], total_chunk_size)
        chunk_sequence_indices = chunk_sequence_indices.flatten() % sequence_length
        indices = indices.unsqueeze(1).expand((indices.shape[0], total_chunk_size, -1)).flatten(0, 1).clone()
        indices[:, -1] = chunk_sequence_indices
        return indices

    def _len_and_dim_norm(self, vectors, sqrt_num):
        """
        length and attention head size dim normalization
        """
        vectors = self._len_norm(vectors)
        vectors = vectors / sqrt_num
        return vectors

    def _len_norm(self, x, epsilon=1e-06):
        """
        length normalization
        """
        variance = torch.mean(x ** 2, -1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + epsilon)
        return norm_x

    def _gather_by_expansion(self, vectors, idxs, num_hashes):
        """
        expand dims of idxs and vectors for all hashes and gather
        """
        expanded_idxs = idxs.unsqueeze(-1).expand(-1, -1, -1, self.attention_head_size)
        vectors = vectors.repeat(1, 1, num_hashes, 1)
        return torch.gather(vectors, 2, expanded_idxs)