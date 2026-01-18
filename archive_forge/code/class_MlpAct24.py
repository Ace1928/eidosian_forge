from typing import Tuple
import torch
import torch.nn.functional as F
from torch import nn
from utils import DTYPE2STR, benchmark_main_helper2, product_dict
import xformers.ops as xops
class MlpAct24(Mlp):

    def fw(self):
        x = self.input
        x = self.fc1(x)
        x = xops.sparsify24(x)
        x = self.act(x)
        x = self.fc2(x)
        self.out = x