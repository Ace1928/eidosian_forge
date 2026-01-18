import math
from typing import Any, Dict, List, Optional, Tuple
import torch
import torchvision
from torch import nn, Tensor
from .image_list import ImageList
from .roi_heads import paste_masks_in_image
@torch.jit.unused
def _get_shape_onnx(image: Tensor) -> Tensor:
    from torch.onnx import operators
    return operators.shape_as_tensor(image)[-2:]