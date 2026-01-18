import torch
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.quant_type import QuantType
from torch.jit._recursive import wrap_cpp_module
def _check_forward_method(model):
    if not model._c._has_method('forward'):
        raise ValueError('input script module does not have forward method')