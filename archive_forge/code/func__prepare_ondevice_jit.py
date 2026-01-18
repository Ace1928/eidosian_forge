import torch
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.quant_type import QuantType
from torch.jit._recursive import wrap_cpp_module
def _prepare_ondevice_jit(model, qconfig_dict, method_name='forward', inplace=False, quant_type=QuantType.STATIC):
    _check_is_script_module(model)
    if not all((isinstance(x, str) for x in qconfig_dict.keys())):
        raise ValueError('qconfig_dict should only contain names(str) as keys.')
    scripted_qconfig_dict = script_qconfig_dict(qconfig_dict)
    method_graph = model._c._get_method(method_name).graph
    torch._C._jit_pass_inline(method_graph)
    model = fuse_conv_bn_jit(model, inplace)
    model_c = torch._C._jit_pass_insert_observer_method_for_ondevice_ptq(model._c, method_name, scripted_qconfig_dict, inplace, quant_type)
    if inplace:
        model._reconstruct(model_c)
    else:
        model = wrap_cpp_module(model_c)
    return model