import logging
import torch
from xformers import _is_triton_available
from xformers.ops import masked_matmul
@classmethod
def _equal(cls, arg0, arg1):
    if not (isinstance(arg0, cls) and isinstance(arg1, cls)):
        return NotImplemented
    if arg0.shape != arg1.shape:
        return False
    if not torch.equal(arg0.__values, arg1.__values):
        return False
    if not torch.equal(arg0.__layout, arg1.__layout):
        return False
    return True