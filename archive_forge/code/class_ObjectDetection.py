from typing import Optional, Tuple, Union
import torch
from torch import nn, Tensor
from . import functional as F, InterpolationMode
class ObjectDetection(nn.Module):

    def forward(self, img: Tensor) -> Tensor:
        if not isinstance(img, Tensor):
            img = F.pil_to_tensor(img)
        return F.convert_image_dtype(img, torch.float)

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'

    def describe(self) -> str:
        return 'Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single ``(C, H, W)`` image ``torch.Tensor`` objects. The images are rescaled to ``[0.0, 1.0]``.'