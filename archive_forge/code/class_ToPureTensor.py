from typing import Any, Dict, Optional, Union
import numpy as np
import PIL.Image
import torch
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F, Transform
from torchvision.transforms.v2._utils import is_pure_tensor
class ToPureTensor(Transform):
    """[BETA] Convert all tv_tensors to pure tensors, removing associated metadata (if any).

    .. v2betastatus:: ToPureTensor transform

    This doesn't scale or change the values, only the type.
    """
    _transformed_types = (tv_tensors.TVTensor,)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> torch.Tensor:
        return inpt.as_subclass(torch.Tensor)