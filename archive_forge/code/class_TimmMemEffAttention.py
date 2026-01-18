import itertools
from functools import partial, reduce
from typing import Iterator
import timm
import torch
import torch.nn as nn
from timm.models.layers import Mlp as TimmMlp
from timm.models.vision_transformer import Attention as TimmAttention
from timm.models.vision_transformer import Block as TimmBlock
from torch.utils import benchmark
import xformers.ops as xops
from xformers.benchmarks.utils import benchmark_main_helper
class TimmMemEffAttention(nn.Module):

    def __init__(self, attn: TimmAttention, op=None):
        super().__init__()
        self.op = None
        self.num_heads = attn.num_heads
        self.scale = attn.scale
        self.qkv = attn.qkv
        self.attn_drop = attn.attn_drop
        self.proj = attn.proj
        self.proj_drop = attn.proj_drop

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = xops.unbind(qkv, dim=2)
        x = xops.memory_efficient_attention(q, k, v, op=self.op).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x