import math
from enum import Enum
from typing import Dict, List, Optional, Tuple
import torch
from torch import Tensor
from . import functional as F, InterpolationMode
@torch.jit.unused
def _tensor_to_pil(self, img: Tensor):
    return F.to_pil_image(img)