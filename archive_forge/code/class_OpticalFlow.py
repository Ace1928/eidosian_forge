from typing import Optional, Tuple, Union
import torch
from torch import nn, Tensor
from . import functional as F, InterpolationMode
class OpticalFlow(nn.Module):

    def forward(self, img1: Tensor, img2: Tensor) -> Tuple[Tensor, Tensor]:
        if not isinstance(img1, Tensor):
            img1 = F.pil_to_tensor(img1)
        if not isinstance(img2, Tensor):
            img2 = F.pil_to_tensor(img2)
        img1 = F.convert_image_dtype(img1, torch.float)
        img2 = F.convert_image_dtype(img2, torch.float)
        img1 = F.normalize(img1, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        img2 = F.normalize(img2, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        img1 = img1.contiguous()
        img2 = img2.contiguous()
        return (img1, img2)

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'

    def describe(self) -> str:
        return 'Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single ``(C, H, W)`` image ``torch.Tensor`` objects. The images are rescaled to ``[-1.0, 1.0]``.'