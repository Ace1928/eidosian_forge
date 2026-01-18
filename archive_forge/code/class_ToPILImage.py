from typing import Any, Dict, Optional, Union
import numpy as np
import PIL.Image
import torch
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F, Transform
from torchvision.transforms.v2._utils import is_pure_tensor
class ToPILImage(Transform):
    """[BETA] Convert a tensor or an ndarray to PIL Image

    .. v2betastatus:: ToPILImage transform

    This transform does not support torchscript.

    Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
    H x W x C to a PIL Image while adjusting the value range depending on the ``mode``.

    Args:
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).
            If ``mode`` is ``None`` (default) there are some assumptions made about the input data:

            - If the input has 4 channels, the ``mode`` is assumed to be ``RGBA``.
            - If the input has 3 channels, the ``mode`` is assumed to be ``RGB``.
            - If the input has 2 channels, the ``mode`` is assumed to be ``LA``.
            - If the input has 1 channel, the ``mode`` is determined by the data type (i.e ``int``, ``float``,
              ``short``).

    .. _PIL.Image mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes
    """
    _transformed_types = (is_pure_tensor, tv_tensors.Image, np.ndarray)

    def __init__(self, mode: Optional[str]=None) -> None:
        super().__init__()
        self.mode = mode

    def _transform(self, inpt: Union[torch.Tensor, PIL.Image.Image, np.ndarray], params: Dict[str, Any]) -> PIL.Image.Image:
        return F.to_pil_image(inpt, mode=self.mode)