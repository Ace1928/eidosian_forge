from typing import Tuple
import torch
import torch.nn.functional as F
from torch import nn
from utils import DTYPE2STR, benchmark_main_helper2, product_dict
import xformers.ops as xops
class LinearW24(torch.nn.Linear):

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w_sparse = xops.sparsify24(self.weight, gradient='24dense', backend='cusparselt')
        return F.linear(input, w_sparse, self.bias)