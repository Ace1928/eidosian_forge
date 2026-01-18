import math
import os
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm as FusedLayerNorm
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, logging
from ...utils.logging import tqdm
from .configuration_jukebox import ATTENTION_PATTERNS, JukeboxConfig, JukeboxPriorConfig, JukeboxVQVAEConfig
class JukeboxAttention(nn.Module):

    def __init__(self, config, n_ctx, attn_func='dense_attn'):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.n_heads = config.n_heads
        self.dropout = config.attn_dropout
        hidden_dim = int(config.attention_multiplier * self.embed_dim)
        self.head_dim = hidden_dim // config.n_heads
        self.n_ctx = n_ctx
        self.hidden_dim = hidden_dim
        self.scale = self.head_dim ** (-0.25)
        self.mask = config.mask
        if attn_func == 'cross_attention':
            self.c_attn = JukeboxConv1D(self.embed_dim, hidden_dim)
            self.c_enc_kv = JukeboxConv1D(self.embed_dim, hidden_dim * 2)
        else:
            self.c_attn = JukeboxConv1D(self.embed_dim, hidden_dim * 3)
        self.c_proj = JukeboxConv1D(hidden_dim, self.embed_dim)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.resid_dropout = nn.Dropout(config.resid_dropout)
        self.attn_func = attn_func
        if attn_func == 'cross_attention':
            self.qkv = self.decode_qkv
        elif attn_func == 'prime_attn':
            self.qkv = self.prime_qkv
        else:
            self.qkv = self.factored_qkv
        ATTENTION_MAP = {'dense_attn': (self.dense_attn, 'autoregressive'), 'block_attn': (self.block_attn, 'autoregressive'), 'transpose_block_attn': (self.transpose_block_attn, 'autoregressive'), 'prev_block_attn': (self.prev_block_attn, None), 'summary_attn': (self.summary_attn, 'summary'), 'summary_spread_attn': (self.summary_spread_attn, 'summary'), 'cross_attention': (self.dense_attn, None), 'prime_attn': (self.prime_attn, 'prime')}
        self.attn, self.attn_mask = ATTENTION_MAP[attn_func]
        self.blocks = config.blocks
        self.spread = config.spread
        if self.blocks is not None:
            self.block_ctx = self.n_ctx // self.blocks
        self.sample_t = 0
        self.cache = {}
        self.encoder_len = config.nb_relevant_lyric_tokens
        self.record_attn = False

    def _attn(self, query_states, key_states, value_states, sample):
        scale = self.scale
        if self.training:
            attention_weight = torch.matmul(query_states * scale, key_states * scale)
        else:
            attention_weight = torch.matmul(query_states, key_states)
            attention_weight.mul_(scale * scale)
        attn_weight_type = attention_weight.dtype
        attention_weight = attention_weight.float()
        if self.mask:
            mask = get_mask(self.attn_mask, query_states.size(-2), key_states.size(-1), self.blocks, self.spread, attention_weight.device, sample, self.sample_t)
            if mask is not None:
                attention_weight = attention_weight * mask + -1000000000.0 * (1 - mask)
        attention_prob = F.softmax(attention_weight, dim=-1).type(attn_weight_type)
        if self.record_attn:
            self.attention_prob = attention_prob
            if self.attn_func == 'prime_attn':
                self.attention_prob = self.attention_prob[:, :, self.encoder_len:, :self.encoder_len]
        attention_prob = self.attn_dropout(attention_prob)
        context_states = torch.matmul(attention_prob, value_states)
        return context_states

    def merge_heads(self, hidden_states):
        hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
        new_hidden_states_shape = (*hidden_states.size()[:-2], hidden_states.size(-2) * hidden_states.size(-1))
        return hidden_states.view(*new_hidden_states_shape)

    def split_heads(self, hidden_states, is_key=False):
        new_hidden_states_shape = (*hidden_states.size()[:-1], self.n_heads, hidden_states.size(-1) // self.n_heads)
        hidden_states = hidden_states.view(*new_hidden_states_shape)
        if is_key:
            return hidden_states.permute(0, 2, 3, 1)
        else:
            return hidden_states.permute(0, 2, 1, 3)

    def dense_attn(self, query, key, value, sample):
        query = self.split_heads(query)
        key = self.split_heads(key, is_key=True)
        value = self.split_heads(value)
        context_states = self._attn(query, key, value, sample)
        context_states = self.merge_heads(context_states)
        return context_states

    def block_attn(self, query, key, value, sample):
        block_ctx = self.block_ctx
        batch_size, seq_len, embed_dim = value.shape
        if sample:
            return self.dense_attn(query, key, value, sample).view(batch_size, 1, embed_dim)
        else:
            query_length = query.shape[1]
            query = query.view(batch_size * query_length // block_ctx, block_ctx, embed_dim)
            if query_length < seq_len:
                seq_len = query_length
                key = key[:, -seq_len:].contiguous()
                value = value[:, -seq_len:].contiguous()
            key = key.view(batch_size * seq_len // block_ctx, block_ctx, embed_dim)
            value = value.view(batch_size * seq_len // block_ctx, block_ctx, embed_dim)
            return self.dense_attn(query, key, value, sample).view(batch_size, seq_len, embed_dim)

    def transpose_block_attn(self, query, key, value, sample):
        block_ctx = self.block_ctx
        batch_size, seq_len, embed_dim = value.shape
        if sample:
            block_len = (seq_len - 1) % block_ctx
            key = key[:, block_len::block_ctx, :]
            value = value[:, block_len::block_ctx, :]
            return self.dense_attn(query, key, value, sample).view(batch_size, 1, embed_dim)
        else:
            query_length = query.shape[1]
            query = query.view(batch_size, query_length // block_ctx, block_ctx, embed_dim)
            query = query.transpose(1, 2).contiguous()
            query = query.view(batch_size * block_ctx, query_length // block_ctx, embed_dim)
            key = key.view(batch_size, seq_len // block_ctx, block_ctx, embed_dim)
            key = key.transpose(1, 2).contiguous()
            key = key.view(batch_size * block_ctx, seq_len // block_ctx, embed_dim)
            value = value.view(batch_size, seq_len // block_ctx, block_ctx, embed_dim)
            value = value.transpose(1, 2).contiguous()
            value = value.view(batch_size * block_ctx, seq_len // block_ctx, embed_dim)
            block_attn = self.dense_attn(query, key, value, sample)
            block_attn = block_attn.view(batch_size, block_ctx, query_length // block_ctx, embed_dim)
            block_attn = block_attn.transpose(1, 2).contiguous()
            block_attn = block_attn.view(batch_size, query_length, embed_dim)
            return block_attn

    def prev_block_attn(self, query, key, value, sample):
        block_ctx = self.block_ctx
        batch_size, seq_len, embed_dim = value.shape
        if sample:
            block = (seq_len - 1) // block_ctx
            prev_l = (block - 1) * block_ctx
            if block > 0:
                key = key[:, prev_l:prev_l + block_ctx, :]
                value = value[:, prev_l:prev_l + block_ctx, :]
            else:
                key = torch.zeros(batch_size, block_ctx, embed_dim, device=query.device, dtype=query.dtype)
                value = torch.zeros(batch_size, block_ctx, embed_dim, device=query.device, dtype=query.dtype)
            return self.dense_attn(query, key, value, sample).view(batch_size, 1, embed_dim)
        else:
            query_length = query.shape[1]
            query = query.view(batch_size * query_length // block_ctx, block_ctx, embed_dim)
            key = key.view(batch_size, seq_len // block_ctx, block_ctx, embed_dim)[:, :-1, :, :]
            key = torch.nn.functional.pad(key, (0, 0, 0, 0, 1, 0))
            key = key.view(batch_size * seq_len // block_ctx, block_ctx, embed_dim)
            value = value.view(batch_size, seq_len // block_ctx, block_ctx, embed_dim)[:, :-1, :, :]
            value = torch.nn.functional.pad(value, (0, 0, 0, 0, 1, 0))
            value = value.view(batch_size * seq_len // block_ctx, block_ctx, embed_dim)
            if query_length < seq_len:
                nb_query_blocks = query_length // block_ctx
                nb_key_blocks = seq_len // block_ctx
                seq_len = query_length
                key = key.view(batch_size, nb_key_blocks, block_ctx, embed_dim)[:, -nb_query_blocks:]
                key = key.contiguous().view(batch_size * nb_query_blocks, block_ctx, embed_dim)
                value = value.view(batch_size, nb_key_blocks, block_ctx, embed_dim)[:, -nb_query_blocks:]
                value = value.contiguous().view(batch_size * nb_query_blocks, block_ctx, embed_dim)
            return self.dense_attn(query, key, value, sample).view(batch_size, seq_len, embed_dim)

    def summary_attn(self, query, key, value, sample):
        blocks = self.blocks
        block_ctx = self.block_ctx
        batch_size, seq_len, embed_dim = value.shape
        if sample:
            key = key[:, block_ctx - 1:blocks * block_ctx - 1:block_ctx, :]
            key = torch.nn.functional.pad(key, (0, 0, 1, 0))
            value = value[:, block_ctx - 1:blocks * block_ctx - 1:block_ctx, :]
            value = torch.nn.functional.pad(value, (0, 0, 1, 0))
            return self.dense_attn(query, key, value, sample).view(batch_size, 1, embed_dim)
        else:
            key = key.view(batch_size, blocks, seq_len // blocks, embed_dim)[:, :-1, -1, :]
            key = torch.nn.functional.pad(key, (0, 0, 1, 0))
            value = value.view(batch_size, blocks, seq_len // blocks, embed_dim)[:, :-1, -1, :]
            value = torch.nn.functional.pad(value, (0, 0, 1, 0))
            return self.dense_attn(query, key, value, sample).view(batch_size, seq_len, embed_dim)

    def summary_spread_attn(self, query, key, value, sample):
        blocks = self.blocks
        spread = self.spread
        batch_size, seq_len, embed_dim = value.shape
        if sample:
            raise NotImplementedError
        else:
            key = key.view(batch_size, blocks, seq_len // blocks, embed_dim)[:, :-1, -spread:, :]
            key = torch.nn.functional.pad(key, (0, 0, 0, 0, 1, 0)).contiguous()
            key = key.view(batch_size, blocks * spread, embed_dim)
            value = value.view(batch_size, blocks, seq_len // blocks, embed_dim)[:, :-1, -spread:, :]
            value = torch.nn.functional.pad(value, (0, 0, 0, 0, 1, 0)).contiguous()
            value = value.view(batch_size, blocks * spread, embed_dim)
            return self.dense_attn(query, key, value, sample).view(batch_size, seq_len, embed_dim)

    def prime_attn(self, query, key, value, sample):
        encoder_len = self._encoder_len
        key = key[:, :encoder_len]
        value = value[:, :encoder_len]
        return self.dense_attn(query, key, value, sample)

    def factored_qkv(self, hidden_states, last_encoder_hidden_states=None, sample=False):
        curr_ctx = hidden_states.shape[1]
        if last_encoder_hidden_states is not None:
            raise TypeError('last_encoder_hidden_states should be None')
        query, key, value = hidden_states.chunk(3, dim=2)
        if sample:
            self.sample_t += curr_ctx
            key, value = self._append_cache(key, value)
            l_cache = self._suff_cache_len()
            if self._cache_len() > l_cache:
                self._slice_cache(-l_cache)
            if curr_ctx > 1:
                if self.attn_func != 'dense_attn':
                    query = self._pad_to_block_ctx(query, query=True)
                    key = self._pad_to_block_ctx(key)
                    value = self._pad_to_block_ctx(value)
                sample = False
            else:
                key = self.cache['key']
                value = self.cache['value']
        return (query, key, value, sample)

    def prime_qkv(self, hidden_states, last_encoder_hidden_states=None, sample=False):
        curr_ctx = hidden_states.shape[1]
        if last_encoder_hidden_states is not None:
            raise TypeError('last_encoder_hidden_states should be None')
        query, key, value = hidden_states.chunk(3, dim=2)
        if sample:
            if self._cache_len() < self._encoder_len:
                self._append_cache(key, value)
            if self._cache_len() > self._encoder_len:
                self._slice_cache(0, self._encoder_len)
            key, value = (self.cache['key'], self.cache['value'])
            self.sample_t += curr_ctx
        return (query, key, value, sample)

    def decode_qkv(self, hidden_states, last_encoder_hidden_states=None, sample=False):
        curr_ctx = hidden_states.shape[1]
        query = hidden_states
        if sample:
            if self.sample_t == 0:
                self.cache['key'], self.cache['value'] = self.c_enc_kv(last_encoder_hidden_states.type_as(hidden_states)).chunk(2, dim=2)
            key, value = (self.cache['key'], self.cache['value'])
            self.sample_t += curr_ctx
        else:
            key, value = self.c_enc_kv(last_encoder_hidden_states.type_as(hidden_states)).chunk(2, dim=2)
        return (query, key, value, sample)

    def forward(self, hidden_states, last_encoder_hidden_states=None, sample=False):
        curr_ctx = hidden_states.shape[1]
        hidden_states = self.c_attn(hidden_states)
        query, key, value, sample = self.qkv(hidden_states, last_encoder_hidden_states=last_encoder_hidden_states, sample=sample)
        attention_scores = self.attn(query, key, value, sample)
        if attention_scores.shape[1] != curr_ctx:
            offset = self._offset(curr_ctx)
            attention_scores = attention_scores[:, offset:offset + curr_ctx, :].contiguous()
        attention_scores = self.c_proj(attention_scores)
        return self.resid_dropout(attention_scores)

    @property
    def _encoder_len(self):
        encoder_len = self.encoder_len
        encoder_blocks = encoder_len // self.blocks + 1
        return encoder_blocks * self.blocks

    def _offset(self, curr_ctx):
        if self.attn_func == 'dense_attn':
            return 0
        return (self.sample_t - curr_ctx) % self.block_ctx

    def _pad_to_block_ctx(self, hidden_states, query=False):
        seq_len = hidden_states.shape[1]
        offset = self._offset(seq_len) if query else 0
        n_blocks = (seq_len + offset + self.block_ctx - 1) // self.block_ctx
        pad = n_blocks * self.block_ctx - seq_len - offset
        if pad == 0 and offset == 0:
            return hidden_states
        else:
            return F.pad(hidden_states, (0, 0, offset, pad))

    def _cache_len(self):
        return 0 if 'key' not in self.cache else self.cache['key'].shape[1]

    def _suff_cache_len(self):
        """
        Precondition:
            key and value are appended with the current context and self.sample_t reflects the 1-indexed sample
            location in the context.
        """
        previous_block_length = (self.sample_t - 1) % self.block_ctx + 1 + self.block_ctx
        REQUIRED_CACHE_LEN = {'dense_attn': self.sample_t, 'block_attn': (self.sample_t - 1) % self.block_ctx + 1, 'transpose_block_attn': self.sample_t, 'prev_block_attn': self.sample_t if self.sample_t <= self.block_ctx else previous_block_length, 'cross_attn': self.encoder_len, 'prime_attn': min(self.sample_t, self._encoder_len)}
        return REQUIRED_CACHE_LEN[self.attn_func]

    def _slice_cache(self, start, end=None):
        self.cache['key'] = self.cache['key'][:, start:end]
        self.cache['value'] = self.cache['value'][:, start:end]

    def _append_cache(self, key, value):
        if 'key' not in self.cache:
            self.cache['key'] = key
            self.cache['value'] = value
        else:
            old_key, old_value = (key, value)
            key = torch.cat([self.cache['key'], old_key], dim=1)
            value = torch.cat([self.cache['value'], old_value], dim=1)
            del self.cache['key']
            del self.cache['value']
            del old_key
            del old_value
            self.cache['key'] = key
            self.cache['value'] = value
        return (self.cache['key'], self.cache['value'])

    def del_cache(self):
        self.sample_t = 0
        if 'key' in self.cache:
            del self.cache['key']
        if 'value' in self.cache:
            del self.cache['value']
        self.cache = {}