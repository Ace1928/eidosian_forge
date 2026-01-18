import torch
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.quant_type import QuantType
from torch.jit._recursive import wrap_cpp_module
def _quantize_jit(model, qconfig_dict, run_fn=None, run_args=None, inplace=False, debug=False, quant_type=QuantType.STATIC):
    if quant_type == QuantType.DYNAMIC:
        model = prepare_dynamic_jit(model, qconfig_dict, inplace)
        model = convert_dynamic_jit(model, True, debug)
    else:
        assert run_fn, 'Must provide calibration function for post training static quantization'
        assert run_args, 'Must provide calibration dataset for post training static quantization'
        model = prepare_jit(model, qconfig_dict, inplace)
        run_fn(model, *run_args)
        model = convert_jit(model, True, debug)
    torch._C._jit_pass_constant_propagation(model.graph)
    torch._C._jit_pass_dce(model.graph)
    return model