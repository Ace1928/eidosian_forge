import os
from typing import List, Optional
import torch
import torch.multiprocessing.reductions
from torch.utils._pytree import tree_flatten, tree_unflatten
from typing_extensions import Annotated
from .. import _is_triton_available
from .common import Alias, make_pytorch_operator_for_dispatch_key
def _should_use_triton(device: torch.device, dtype: torch.dtype) -> bool:
    if not int(os.getenv('XFORMERS_TILED_MATMUL_ENABLE_TRITON', '1')):
        return False
    if not TRITON_IS_AVAILABLE:
        return False
    device_capability = torch.cuda.get_device_capability(device)
    if device_capability < (8, 0):
        return False
    return True