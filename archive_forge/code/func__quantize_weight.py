import torch
import typing
def _quantize_weight(weight: torch.Tensor, weight_qscheme: torch.qscheme, weight_dtype: torch.dtype, weight_scale: torch.Tensor, weight_zero_point: torch.Tensor, weight_axis_int: int) -> torch.Tensor:
    if weight_dtype == torch.float16:
        weight = weight.to(weight_dtype)
        return weight
    if weight_qscheme == torch.per_tensor_affine:
        if weight_dtype in [torch.quint8, torch.qint8, torch.qint32]:
            weight = torch.quantize_per_tensor(weight, weight_scale, weight_zero_point, weight_dtype)
            return weight
    elif weight_qscheme in [torch.per_channel_affine, torch.per_channel_affine_float_qparams]:
        if weight_dtype in [torch.quint8, torch.qint8, torch.quint4x2, torch.qint32]:
            weight = torch.quantize_per_channel(weight, weight_scale, weight_zero_point, weight_axis_int, weight_dtype)
            return weight
    raise Exception(f'Unsupported dtype and qscheme: {weight_dtype}, {weight_qscheme}')