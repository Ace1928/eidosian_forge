import sys
from typing import Any, Dict, Type
import torch
from torch.utils import benchmark
from utils import benchmark_main_helper2
import xformers.ops as xops
class AttentionDecodingFlashAttention(AttentionDecodingFlashDecoding):

    def fw(self) -> None:
        q, k, v = (self.q, self.k, self.v)
        if q.ndim == 5:
            B, Mq, H1, H2, K = q.shape
            B, Mkv, H1, H2, K = k.shape
            q = q.reshape([B, Mq, H1 * H2, K])
            k = k[:, :, :, 0]
            v = v[:, :, :, 0]
        return flash_attn.flash_attn_func(q, k, v)