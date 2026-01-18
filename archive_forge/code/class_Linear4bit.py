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
class Linear4bit(nn.Linear):
    """
    This class is the base module for the 4-bit quantization algorithm presented in [QLoRA](https://arxiv.org/abs/2305.14314).
    QLoRA 4-bit linear layers uses blockwise k-bit quantization under the hood, with the possibility of selecting various
    compute datatypes such as FP4 and NF4.

    In order to quantize a linear layer one should first load the original fp16 / bf16 weights into
    the Linear4bit module, then call `quantized_module.to("cuda")` to quantize the fp16 / bf16 weights.

    Example:

    ```python
    import torch
    import torch.nn as nn

    import bitsandbytes as bnb
    from bnb.nn import Linear4bit

    fp16_model = nn.Sequential(
        nn.Linear(64, 64),
        nn.Linear(64, 64)
    )

    quantized_model = nn.Sequential(
        Linear4bit(64, 64),
        Linear4bit(64, 64)
    )

    quantized_model.load_state_dict(fp16_model.state_dict())
    quantized_model = quantized_model.to(0) # Quantization happens here
    ```
    """

    def __init__(self, input_features, output_features, bias=True, compute_dtype=None, compress_statistics=True, quant_type='fp4', quant_storage=torch.uint8, device=None):
        """
        Initialize Linear4bit class.

        Args:
            input_features (`str`):
                Number of input features of the linear layer.
            output_features (`str`):
                Number of output features of the linear layer.
            bias (`bool`, defaults to `True`):
                Whether the linear class uses the bias term as well.
        """
        super().__init__(input_features, output_features, bias, device)
        self.weight = Params4bit(self.weight.data, requires_grad=False, compress_statistics=compress_statistics, quant_type=quant_type, quant_storage=quant_storage, module=self)
        self.compute_dtype = compute_dtype
        self.compute_type_is_set = False
        self.quant_state = None
        self.quant_storage = quant_storage

    def set_compute_type(self, x):
        if x.dtype in [torch.float32, torch.bfloat16]:
            self.compute_dtype = x.dtype
        elif x.dtype == torch.float16:
            if self.compute_dtype == torch.float32 and x.numel() == x.shape[-1]:
                warnings.warn('Input type into Linear4bit is torch.float16, but bnb_4bit_compute_dtype=torch.float32 (default). This will lead to slow inference.')
                warnings.filterwarnings('ignore', message='.*inference.')
            if self.compute_dtype == torch.float32 and x.numel() != x.shape[-1]:
                warnings.warn('Input type into Linear4bit is torch.float16, but bnb_4bit_compute_dtype=torch.float32 (default). This will lead to slow inference or training speed.')
                warnings.filterwarnings('ignore', message='.*inference or training')

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """
        save weight and bias,
        then fill state_dict with components of quant_state
        """
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if getattr(self.weight, 'quant_state', None) is not None:
            for k, v in self.weight.quant_state.as_dict(packed=True).items():
                destination[prefix + 'weight.' + k] = v if keep_vars else v.detach()

    def forward(self, x: torch.Tensor):
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)
        if getattr(self.weight, 'quant_state', None) is None:
            if getattr(self, 'quant_state', None) is not None:
                assert self.weight.shape[1] == 1
                if not isinstance(self.weight, Params4bit):
                    self.weight = Params4bit(self.weight, quant_storage=self.quant_storage)
                self.weight.quant_state = self.quant_state
            else:
                print('FP4 quantization state not initialized. Please call .cuda() or .to(device) on the LinearFP4 layer first.')
        if not self.compute_type_is_set:
            self.set_compute_type(x)
            self.compute_type_is_set = True
        inp_dtype = x.dtype
        if self.compute_dtype is not None:
            x = x.to(self.compute_dtype)
        bias = None if self.bias is None else self.bias.to(self.compute_dtype)
        out = bnb.matmul_4bit(x, self.weight.t(), bias=bias, quant_state=self.weight.quant_state)
        out = out.to(inp_dtype)
        return out