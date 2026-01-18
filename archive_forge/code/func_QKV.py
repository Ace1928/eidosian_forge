import functools
import os, math, gc, importlib
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy
from torch.utils.cpp_extension import load
@MyFunction
def QKV(self, q, k, v):
    att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(self.att_mask == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    x = att @ v
    return x