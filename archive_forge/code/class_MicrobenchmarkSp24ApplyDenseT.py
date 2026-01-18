from typing import Tuple
import torch
import torch.nn.functional as F
from torch import nn
from utils import DTYPE2STR, benchmark_main_helper2, product_dict
import xformers.ops as xops
class MicrobenchmarkSp24ApplyDenseT(MicrobenchmarkBase):

    def fw(self) -> torch.Tensor:
        xops.sparsify24_like(self.input_colMajor, pattern=self.input_sp, out_dense=True)
        return self.input