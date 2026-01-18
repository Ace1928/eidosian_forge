import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple
import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
from torch import nn
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict
from typing import Optional
import types, gc, os, time, re
import torch
import torch.nn as nn
from torch.nn import functional as F
class RWKV_6(torch.autograd.Function):

    @staticmethod
    def forward(ctx, B, T, C, H, state, r, k, v, w, u):
        with torch.no_grad():
            assert HEAD_SIZE == C // H
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            assert state.dtype == torch.float32
            assert w.dtype == torch.float32
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w.is_contiguous()
            assert u.is_contiguous()
            eew = torch.exp(-torch.exp(w.float())).contiguous()
            y = torch.empty((B, T, C), device=w.device, dtype=r.dtype, memory_format=torch.contiguous_format)
            if r.dtype == torch.bfloat16:
                rwkv6.forward_bf16(B, T, C, H, state, r, k, v, eew, u, y)
            elif r.dtype == torch.float16:
                rwkv6.forward_fp16(B, T, C, H, state, r, k, v, eew, u, y)
            elif r.dtype == torch.float32:
                rwkv6.forward_fp32(B, T, C, H, state, r, k, v, eew, u, y)
            return (y, state)