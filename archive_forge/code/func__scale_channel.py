import warnings
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.nn.functional import conv2d, grid_sample, interpolate, pad as torch_pad
def _scale_channel(img_chan: Tensor) -> Tensor:
    if img_chan.is_cuda:
        hist = torch.histc(img_chan.to(torch.float32), bins=256, min=0, max=255)
    else:
        hist = torch.bincount(img_chan.reshape(-1), minlength=256)
    nonzero_hist = hist[hist != 0]
    step = torch.div(nonzero_hist[:-1].sum(), 255, rounding_mode='floor')
    if step == 0:
        return img_chan
    lut = torch.div(torch.cumsum(hist, 0) + torch.div(step, 2, rounding_mode='floor'), step, rounding_mode='floor')
    lut = torch.nn.functional.pad(lut, [1, 0])[:-1].clamp(0, 255)
    return lut[img_chan.to(torch.int64)].to(torch.uint8)