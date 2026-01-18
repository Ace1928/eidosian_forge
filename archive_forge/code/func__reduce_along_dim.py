import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
def _reduce_along_dim(self: List[int], dim: int, keepdim: bool):
    dim = maybe_wrap_dim(dim, len(self))
    out: List[int] = []
    for i, self_dim in enumerate(self):
        if i == dim:
            if keepdim:
                out.append(1)
        else:
            out.append(self_dim)
    return out