import sys
from typing import Any, Dict, Type
import torch
from torch.utils import benchmark
from utils import benchmark_main_helper2
import xformers.ops as xops
class AttentionDecodingCK(AttentionDecodingFlashDecoding):
    OP = xops.fmha.ck.FwOp