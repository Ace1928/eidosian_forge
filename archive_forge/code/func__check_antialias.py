import math
import numbers
import warnings
from enum import Enum
from typing import Any, List, Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from ..utils import _log_api_usage_once
from . import _functional_pil as F_pil, _functional_tensor as F_t
def _check_antialias(img: Tensor, antialias: Optional[Union[str, bool]], interpolation: InterpolationMode) -> Optional[bool]:
    if isinstance(antialias, str):
        if isinstance(img, Tensor) and (interpolation == InterpolationMode.BILINEAR or interpolation == InterpolationMode.BICUBIC):
            warnings.warn('The default value of the antialias parameter of all the resizing transforms (Resize(), RandomResizedCrop(), etc.) will change from None to True in v0.17, in order to be consistent across the PIL and Tensor backends. To suppress this warning, directly pass antialias=True (recommended, future default), antialias=None (current default, which means False for Tensors and True for PIL), or antialias=False (only works on Tensors - PIL will still use antialiasing). This also applies if you are using the inference transforms from the models weights: update the call to weights.transforms(antialias=True).')
        antialias = None
    return antialias