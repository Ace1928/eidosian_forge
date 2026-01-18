import copy
from typing import Any, Dict, Optional, TypeVar, Union, overload
import warnings
import torch
from torch import Tensor, device, dtype, nn
import torch.nn.functional as F
import bitsandbytes as bnb
from bitsandbytes.autograd._functions import get_tile_inds, undo_layout
from bitsandbytes.functional import QuantState
from bitsandbytes.optim import GlobalOptimManager
from bitsandbytes.utils import OutlierTracer
class SwitchBackLinearBnb(nn.Linear):

    def __init__(self, input_features, output_features, bias=True, has_fp16_weights=True, memory_efficient_backward=False, threshold=0.0, index=None, device=None):
        super().__init__(input_features, output_features, bias, device)
        self.state = bnb.MatmulLtState()
        self.index = index
        self.state.threshold = threshold
        self.state.has_fp16_weights = has_fp16_weights
        self.state.memory_efficient_backward = memory_efficient_backward
        if threshold > 0.0 and (not has_fp16_weights):
            self.state.use_pool = True
        self.weight = Int8Params(self.weight.data, has_fp16_weights=has_fp16_weights, requires_grad=has_fp16_weights)

    def init_8bit_state(self):
        self.state.CB = self.weight.CB
        self.state.SCB = self.weight.SCB
        self.weight.CB = None
        self.weight.SCB = None

    def forward(self, x):
        self.state.is_training = self.training
        if self.weight.CB is not None:
            self.init_8bit_state()
        out = bnb.matmul_mixed(x.half(), self.weight.half(), bias=None, state=self.state) + self.bias