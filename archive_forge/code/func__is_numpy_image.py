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
@torch.jit.unused
def _is_numpy_image(img: Any) -> bool:
    return img.ndim in {2, 3}