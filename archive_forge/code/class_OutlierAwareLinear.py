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
class OutlierAwareLinear(nn.Linear):

    def __init__(self, input_features, output_features, bias=True, device=None):
        super().__init__(input_features, output_features, bias, device)
        self.outlier_dim = None
        self.is_quantized = False

    def forward_with_outliers(self, x, outlier_idx):
        raise NotImplementedError('Please override the `forward_with_outliers(self, x, outlier_idx)` function')

    def quantize_weight(self, w, outlier_idx):
        raise NotImplementedError('Please override the `quantize_weights(self, w, outlier_idx)` function')

    def forward(self, x):
        if self.outlier_dim is None:
            tracer = OutlierTracer.get_instance()
            if not tracer.is_initialized():
                print('Please use OutlierTracer.initialize(model) before using the OutlierAwareLinear layer')
            outlier_idx = tracer.get_outliers(self.weight)
            self.outlier_dim = outlier_idx
        if not self.is_quantized:
            w = self.quantize_weight(self.weight, self.outlier_dim)
            self.weight.data.copy_(w)
            self.is_quantized = True