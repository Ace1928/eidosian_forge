import math
from typing import Iterable, Iterator, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_graphormer import GraphormerConfig
class GraphormerMultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, config: GraphormerConfig):
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.kdim = config.kdim if config.kdim is not None else config.embedding_dim
        self.vdim = config.vdim if config.vdim is not None else config.embedding_dim
        self.qkv_same_dim = self.kdim == config.embedding_dim and self.vdim == config.embedding_dim
        self.num_heads = config.num_attention_heads
        self.attention_dropout_module = torch.nn.Dropout(p=config.attention_dropout, inplace=False)
        self.head_dim = config.embedding_dim // config.num_attention_heads
        if not self.head_dim * config.num_attention_heads == self.embedding_dim:
            raise AssertionError('The embedding_dim must be divisible by num_heads.')
        self.scaling = self.head_dim ** (-0.5)
        self.self_attention = True
        if not self.self_attention:
            raise NotImplementedError('The Graphormer model only supports self attention for now.')
        if self.self_attention and (not self.qkv_same_dim):
            raise AssertionError('Self-attention requires query, key and value to be of the same size.')
        self.k_proj = quant_noise(nn.Linear(self.kdim, config.embedding_dim, bias=config.bias), config.q_noise, config.qn_block_size)
        self.v_proj = quant_noise(nn.Linear(self.vdim, config.embedding_dim, bias=config.bias), config.q_noise, config.qn_block_size)
        self.q_proj = quant_noise(nn.Linear(config.embedding_dim, config.embedding_dim, bias=config.bias), config.q_noise, config.qn_block_size)
        self.out_proj = quant_noise(nn.Linear(config.embedding_dim, config.embedding_dim, bias=config.bias), config.q_noise, config.qn_block_size)
        self.onnx_trace = False

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, query: torch.LongTensor, key: Optional[torch.Tensor], value: Optional[torch.Tensor], attn_bias: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor]=None, need_weights: bool=True, attn_mask: Optional[torch.Tensor]=None, before_softmax: bool=False, need_head_weights: bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            key_padding_mask (Bytetorch.Tensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (Bytetorch.Tensor, optional): typically used to
                implement causal attention, where the mask prevents the attention from looking forward in time
                (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default: return the average attention weights over all
                heads.
        """
        if need_head_weights:
            need_weights = True
        tgt_len, bsz, embedding_dim = query.size()
        src_len = tgt_len
        if not embedding_dim == self.embedding_dim:
            raise AssertionError(f'The query embedding dimension {embedding_dim} is not equal to the expected embedding_dim {self.embedding_dim}.')
        if not list(query.size()) == [tgt_len, bsz, embedding_dim]:
            raise AssertionError('Query size incorrect in Graphormer, compared to model dimensions.')
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                if key_bsz != bsz or value is None or (not (src_len, bsz == value.shape[:2])):
                    raise AssertionError('The batch shape does not match the key or value shapes provided to the attention.')
        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)
        q *= self.scaling
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is None or not k.size(1) == src_len:
            raise AssertionError('The shape of the key generated in the attention is incorrect')
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        if key_padding_mask is not None:
            if key_padding_mask.size(0) != bsz or key_padding_mask.size(1) != src_len:
                raise AssertionError('The shape of the generated padding mask for the key does not match expected dimensions.')
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)
        if list(attn_weights.size()) != [bsz * self.num_heads, tgt_len, src_len]:
            raise AssertionError('The attention weights generated do not match the expected dimensions.')
        if attn_bias is not None:
            attn_weights += attn_bias.view(bsz * self.num_heads, tgt_len, src_len)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float('-inf'))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if before_softmax:
            return (attn_weights, v)
        attn_weights_float = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.attention_dropout_module(attn_weights)
        if v is None:
            raise AssertionError('No value generated')
        attn = torch.bmm(attn_probs, v)
        if list(attn.size()) != [bsz * self.num_heads, tgt_len, self.head_dim]:
            raise AssertionError('The attention generated do not match the expected dimensions.')
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embedding_dim)
        attn: torch.Tensor = self.out_proj(attn)
        attn_weights = None
        if need_weights:
            attn_weights = attn_weights_float.contiguous().view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
            if not need_head_weights:
                attn_weights = attn_weights.mean(dim=0)
        return (attn, attn_weights)

    def apply_sparse_mask(self, attn_weights: torch.Tensor, tgt_len: int, src_len: int, bsz: int) -> torch.Tensor:
        return attn_weights