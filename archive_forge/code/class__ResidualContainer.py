import math
from typing import List, Optional, Tuple
import torch
from torchaudio.models.emformer import _EmformerAttention, _EmformerImpl, _get_weight_init_gains
class _ResidualContainer(torch.nn.Module):

    def __init__(self, module: torch.nn.Module, output_weight: int):
        super().__init__()
        self.module = module
        self.output_weight = output_weight

    def forward(self, input: torch.Tensor):
        output = self.module(input)
        return output * self.output_weight + input