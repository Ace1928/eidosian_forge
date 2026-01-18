import math
from typing import List, Optional
import torch
from torch import nn, Tensor
from .image_list import ImageList
def _grid_default_boxes(self, grid_sizes: List[List[int]], image_size: List[int], dtype: torch.dtype=torch.float32) -> Tensor:
    default_boxes = []
    for k, f_k in enumerate(grid_sizes):
        if self.steps is not None:
            x_f_k = image_size[1] / self.steps[k]
            y_f_k = image_size[0] / self.steps[k]
        else:
            y_f_k, x_f_k = f_k
        shifts_x = ((torch.arange(0, f_k[1]) + 0.5) / x_f_k).to(dtype=dtype)
        shifts_y = ((torch.arange(0, f_k[0]) + 0.5) / y_f_k).to(dtype=dtype)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        shifts = torch.stack((shift_x, shift_y) * len(self._wh_pairs[k]), dim=-1).reshape(-1, 2)
        _wh_pair = self._wh_pairs[k].clamp(min=0, max=1) if self.clip else self._wh_pairs[k]
        wh_pairs = _wh_pair.repeat(f_k[0] * f_k[1], 1)
        default_box = torch.cat((shifts, wh_pairs), dim=1)
        default_boxes.append(default_box)
    return torch.cat(default_boxes, dim=0)