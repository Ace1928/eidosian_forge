import torch
import torch.distributed as dist
from torch import nn
def _quantize_per_channel_cuda(x, scale, zero_point):
    y = torch.zeros(x.size(), device=x.device)
    for i in range(x.size()[0]):
        y[i, :] = torch.round(x[i, :] / scale[i]) + zero_point[i]
    y = torch.clamp(y, 0, 255).to(torch.uint8)
    return y