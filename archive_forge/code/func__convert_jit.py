import torch
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.quant_type import QuantType
from torch.jit._recursive import wrap_cpp_module
def _convert_jit(model, inplace=False, debug=False, quant_type=QuantType.STATIC, preserved_attrs=None):
    _check_is_script_module(model)
    model.eval()
    model_c = model._c
    model_c = torch._C._jit_pass_insert_quant_dequant(model_c, 'forward', inplace, debug, quant_type)
    if not debug:
        is_xpu = all((p.device.type == 'xpu' for p in model.parameters()))
        if not is_xpu:
            model.cpu()
        if preserved_attrs is None:
            preserved_attrs = []
        model_c = torch._C._jit_pass_quant_finalize(model_c, quant_type, preserved_attrs)
    if inplace:
        model._reconstruct(model_c)
    else:
        model = wrap_cpp_module(model_c)
    torch._C._jit_pass_constant_propagation(model.graph)
    torch._C._jit_pass_dce(model.graph)
    return model