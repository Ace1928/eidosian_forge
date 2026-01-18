import torch
import typing
def _get_weight_qparam_keys(state_dict: typing.Dict[str, typing.Any], prefix: str):
    keys = ['weight_qscheme', 'weight_dtype']
    weight_qscheme = state_dict[prefix + 'weight_qscheme']
    if weight_qscheme is not None:
        keys.append('weight_scale')
        keys.append('weight_zero_point')
        if weight_qscheme == torch.quantize_per_channel:
            keys.append('weight_axis')
    return keys