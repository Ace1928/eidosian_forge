from typing import Optional, Tuple, Union
import torch
from torch import nn, Tensor
from . import functional as F, InterpolationMode
class SemanticSegmentation(nn.Module):

    def __init__(self, *, resize_size: Optional[int], mean: Tuple[float, ...]=(0.485, 0.456, 0.406), std: Tuple[float, ...]=(0.229, 0.224, 0.225), interpolation: InterpolationMode=InterpolationMode.BILINEAR, antialias: Optional[Union[str, bool]]='warn') -> None:
        super().__init__()
        self.resize_size = [resize_size] if resize_size is not None else None
        self.mean = list(mean)
        self.std = list(std)
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img: Tensor) -> Tensor:
        if isinstance(self.resize_size, list):
            img = F.resize(img, self.resize_size, interpolation=self.interpolation, antialias=self.antialias)
        if not isinstance(img, Tensor):
            img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        format_string += f'\n    resize_size={self.resize_size}'
        format_string += f'\n    mean={self.mean}'
        format_string += f'\n    std={self.std}'
        format_string += f'\n    interpolation={self.interpolation}'
        format_string += '\n)'
        return format_string

    def describe(self) -> str:
        return f'Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single ``(C, H, W)`` image ``torch.Tensor`` objects. The images are resized to ``resize_size={self.resize_size}`` using ``interpolation={self.interpolation}``. Finally the values are first rescaled to ``[0.0, 1.0]`` and then normalized using ``mean={self.mean}`` and ``std={self.std}``.'