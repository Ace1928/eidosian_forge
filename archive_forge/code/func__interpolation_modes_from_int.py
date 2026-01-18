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
def _interpolation_modes_from_int(i: int) -> InterpolationMode:
    inverse_modes_mapping = {0: InterpolationMode.NEAREST, 2: InterpolationMode.BILINEAR, 3: InterpolationMode.BICUBIC, 4: InterpolationMode.BOX, 5: InterpolationMode.HAMMING, 1: InterpolationMode.LANCZOS}
    return inverse_modes_mapping[i]