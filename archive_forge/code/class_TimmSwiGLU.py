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
class TimmSwiGLU(nn.Module):

    def __init__(self, mlp: TimmMlp, op=None) -> None:
        super().__init__()
        self.fc1 = mlp.fc1
        self.swiglu = xops.SwiGLU(in_features=mlp.fc1.in_features, hidden_features=mlp.fc1.out_features, bias=True)
        self.op = op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swiglu(x)