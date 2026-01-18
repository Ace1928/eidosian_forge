import math
import hydra
import torch
import torch.nn as nn
from einops import rearrange
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
from flash_attn.flash_blocksparse_attn_interface import (
class FlashBlocksparseMHA(nn.Module):

    def __init__(self, embed_dim, num_heads, sparsity_config, bias=True, batch_first=True, attention_dropout=0.0, causal=False, max_seq_length=2048, device=None, dtype=None, **kwargs) -> None:
        assert batch_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, 'self.kdim must be divisible by num_heads'
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim in [16, 32, 64], 'Only support head_dim == 16, 32, or 64'
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.inner_attn = FlashBlocksparseAttention(sparsity_config, attention_dropout=attention_dropout, max_seq_length=max_seq_length, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(self, x, x_ignored_, x_ignored_1_, attn_mask=None, key_padding_mask=None, need_weights=False):
        qkv = self.Wqkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)
        context, attn_weights = self.inner_attn(qkv, key_padding_mask=key_padding_mask, need_weights=need_weights, causal=self.causal)
        return (self.out_proj(rearrange(context, 'b s h d -> b s (h d)')), attn_weights)