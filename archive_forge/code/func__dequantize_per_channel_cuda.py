import torch
import torch.distributed as dist
from torch import nn
def _dequantize_per_channel_cuda(y, scale, zero_point):
    y = y.to(torch.float32).cuda(y.device)
    x = torch.zeros_like(y, device=y.device)
    for i in range(x.size()[0]):
        x[i, :] = scale[i] * (y[i, :] - zero_point[i])
    return x