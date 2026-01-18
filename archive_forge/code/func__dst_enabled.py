from enum import Enum
from typing import Optional, Tuple
import torch
from torch import Tensor
@property
def _dst_enabled(self) -> bool:
    """True if DST is enabled."""
    return self._dst_top_k_element is not None or self._dst_top_k_percent is not None