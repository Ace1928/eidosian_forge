import numpy as np
import torch
from contextlib import contextmanager
from torch.testing._internal.common_utils import TEST_WITH_ASAN, TEST_WITH_TSAN, TEST_WITH_UBSAN, IS_PPC, IS_MACOS, IS_WINDOWS
def _fake_quantize_per_channel_affine_grad_reference(dY, X, per_channel_scale, per_channel_zero_point, axis, quant_min, quant_max):
    dtype = X.dtype
    X, permute_axis_list = _permute_to_axis_zero(X.to(torch.float32), axis)
    Xq = torch.zeros_like(X)
    for i in range(X.size()[0]):
        Xq[i] = torch.round(X[i] * (1.0 / per_channel_scale[i]) + per_channel_zero_point[i])
    Xq = Xq.permute(tuple(permute_axis_list))
    mask = (Xq >= quant_min) * (Xq <= quant_max)
    res = torch.zeros_like(dY)
    res[mask] = dY[mask]
    return res.to(dtype)