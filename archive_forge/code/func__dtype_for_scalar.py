from collections import namedtuple
import torch
from . import _casting_dicts as _cd
def _dtype_for_scalar(py_type):
    return {bool: torch.bool, torch.SymBool: torch.bool, int: torch.int64, torch.SymInt: torch.int64, float: torch.float64, torch.SymFloat: torch.float64, complex: torch.complex128}[py_type]