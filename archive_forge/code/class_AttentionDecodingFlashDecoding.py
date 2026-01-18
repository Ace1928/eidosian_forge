import sys
from typing import Any, Dict, Type
import torch
from torch.utils import benchmark
from utils import benchmark_main_helper2
import xformers.ops as xops
class AttentionDecodingFlashDecoding:
    OP: Any = xops.fmha.flash.FwOp

    def __init__(self, B: int, Mq: int, Mkv: int, Hq: int, Hkv: int, K: int, bw: bool) -> None:
        dtype = torch.float16
        self.sub_label = f'B={B} Mq={Mq} Mkv={Mkv} Hq={Hq} Hkv={Hkv} K={K} TotalBytes={(B * Mkv * Hkv * K * 2 + B * Mq * Hq * K + B * Mq * Hq * K) * 2}'
        self.label = 'attn_decoding'
        self.shapes = (B, Mq, Mkv, Hq, Hkv, K)
        assert Hkv <= Hq
        assert Hq % Hkv == 0
        self.q = torch.randn([B, Mq, Hkv, Hq // Hkv, K], device='cuda', dtype=dtype, requires_grad=bw)
        self.k = torch.randn([B, Mkv, Hkv, 1, K], device='cuda', dtype=dtype, requires_grad=bw).expand(-1, -1, -1, Hq // Hkv, -1)
        self.v = torch.randn([B, Mkv, Hkv, 1, K], device='cuda', dtype=dtype, requires_grad=bw).expand(-1, -1, -1, Hq // Hkv, -1)
        if Hq == Hkv:
            self.q = self.q[:, :, :, 0]
            self.k = self.k[:, :, :, 0]
            self.v = self.v[:, :, :, 0]
        if Hkv == 1:
            self.q = self.q[:, :, 0]
            self.k = self.k[:, :, 0]
            self.v = self.v[:, :, 0]

    def fw(self) -> None:
        try:
            xops.memory_efficient_attention_forward(self.q, self.k, self.v, op=self.OP)
        except (RuntimeError, ValueError) as e:
            print(f'Runtime error: {e}')