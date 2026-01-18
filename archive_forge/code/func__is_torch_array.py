from __future__ import annotations
import sys
import math
def _is_torch_array(x):
    if 'torch' not in sys.modules:
        return False
    import torch
    return isinstance(x, torch.Tensor)