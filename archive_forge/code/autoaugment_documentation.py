import math
from enum import Enum
from typing import Dict, List, Optional, Tuple
import torch
from torch import Tensor
from . import functional as F, InterpolationMode

            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        