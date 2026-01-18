import math
from typing import Any, Dict, List, Optional, Tuple
import torch
import torchvision
from torch import nn, Tensor
from .image_list import ImageList
from .roi_heads import paste_masks_in_image
def _resize_image_and_masks(image: Tensor, self_min_size: int, self_max_size: int, target: Optional[Dict[str, Tensor]]=None, fixed_size: Optional[Tuple[int, int]]=None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
    if torchvision._is_tracing():
        im_shape = _get_shape_onnx(image)
    else:
        im_shape = torch.tensor(image.shape[-2:])
    size: Optional[List[int]] = None
    scale_factor: Optional[float] = None
    recompute_scale_factor: Optional[bool] = None
    if fixed_size is not None:
        size = [fixed_size[1], fixed_size[0]]
    else:
        if torch.jit.is_scripting() or torchvision._is_tracing():
            min_size = torch.min(im_shape).to(dtype=torch.float32)
            max_size = torch.max(im_shape).to(dtype=torch.float32)
            self_min_size_f = float(self_min_size)
            self_max_size_f = float(self_max_size)
            scale = torch.min(self_min_size_f / min_size, self_max_size_f / max_size)
            if torchvision._is_tracing():
                scale_factor = _fake_cast_onnx(scale)
            else:
                scale_factor = scale.item()
        else:
            min_size = min(im_shape)
            max_size = max(im_shape)
            scale_factor = min(self_min_size / min_size, self_max_size / max_size)
        recompute_scale_factor = True
    image = torch.nn.functional.interpolate(image[None], size=size, scale_factor=scale_factor, mode='bilinear', recompute_scale_factor=recompute_scale_factor, align_corners=False)[0]
    if target is None:
        return (image, target)
    if 'masks' in target:
        mask = target['masks']
        mask = torch.nn.functional.interpolate(mask[:, None].float(), size=size, scale_factor=scale_factor, recompute_scale_factor=recompute_scale_factor)[:, 0].byte()
        target['masks'] = mask
    return (image, target)