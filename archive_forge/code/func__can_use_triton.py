import logging
import torch
from xformers import _is_triton_available
from xformers.ops import masked_matmul
def _can_use_triton(a):
    if a.device.type == 'cpu':
        return False
    if blocksparse_matmul is None:
        return False
    return True