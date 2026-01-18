import sys
from typing import Any, Dict, Type
import torch
from torch.utils import benchmark
from utils import benchmark_main_helper2
import xformers.ops as xops
class AttentionDecodingPyTorchRepeat(AttentionDecodingFlashDecoding):

    def fw(self) -> None:
        B, Mq, Mkv, Hq, Hkv, K = self.shapes
        scale = 1 / K ** 0.5
        q = self.q.reshape([B, Mq, -1, K]).permute(0, 2, 1, 3)
        k = self.k.reshape([B, Mkv, -1, K]).permute(0, 2, 1, 3)
        v = self.v.reshape([B, Mkv, -1, K]).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-1, -2)).softmax(-1) * scale
        return attn @ v