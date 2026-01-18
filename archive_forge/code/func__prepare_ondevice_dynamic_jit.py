import torch
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.quant_type import QuantType
from torch.jit._recursive import wrap_cpp_module
def _prepare_ondevice_dynamic_jit(model, qconfig_dict, method_name='forward', inplace=False):
    return _prepare_ondevice_jit(model, qconfig_dict, method_name, inplace, quant_type=QuantType.DYNAMIC)