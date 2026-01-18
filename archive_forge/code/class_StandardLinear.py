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
class StandardLinear(nn.Linear):

    def forward(self, x):
        return StandardLinearFunction.apply(x, self.weight, self.bias)