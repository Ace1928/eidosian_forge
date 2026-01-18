from functools import partial
import torch
import torch.nn as nn
from bitsandbytes.triton.dequantize_rowwise import dequantize_rowwise
from bitsandbytes.triton.int8_matmul_mixed_dequantize import (
from bitsandbytes.triton.int8_matmul_rowwise_dequantize import (
from bitsandbytes.triton.quantize_columnwise_and_transpose import (
from bitsandbytes.triton.quantize_global import (
from bitsandbytes.triton.quantize_rowwise import quantize_rowwise
from bitsandbytes.triton.triton_utils import is_triton_available
class SwitchBackLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool=True, device=None, dtype=None, vector_wise_quantization: bool=False, mem_efficient: bool=False):
        super().__init__(in_features, out_features, bias, device, dtype)
        if not is_triton_available():
            raise ImportError('Could not import triton. Please install triton to use SwitchBackLinear.\n                               Alternatively, you can use bnb.nn.SwitchBackLinearBnb, but it will be slower')
        self.vector_wise_quantization = vector_wise_quantization
        if self.vector_wise_quantization:
            self._fn = _switchback_vectorrize
            if mem_efficient:
                print('mem efficient is not supported for vector-wise quantization.')
                exit(1)
        elif mem_efficient:
            self._fn = _switchback_global_mem_efficient
        else:
            self._fn = _switchback_global

    def prepare_for_eval(self):
        print('=> preparing for eval.')
        if self.vector_wise_quantization:
            W_int8, state_W = quantize_rowwise(self.weight)
        else:
            W_int8, state_W = quantize_global(self.weight)
        self.register_buffer('W_int8', W_int8)
        self.register_buffer('state_W', state_W)
        del self.weight

    def forward(self, x):
        if self.training:
            return self._fn.apply(x, self.weight, self.bias)
        else:
            if not hasattr(self, 'W_int8'):
                return self._fn.apply(x, self.weight, self.bias)
            X = x.view(-1, x.size(-1))
            X_int8, state_X = quantize_rowwise(X)
            if self.vector_wise_quantization:
                return int8_matmul_rowwise_dequantize(X_int8, self.W_int8.t(), state_X, self.state_W, self.bias).view(*x.size()[:-1], -1)
            else:
                return int8_matmul_mixed_dequantize(X_int8, self.W_int8.t(), state_X, self.state_W, self.bias).view(*x.size()[:-1], -1)