import builtins
import torch
from . import _dtypes_impl
class float64(floating):
    name = 'float64'
    typecode = 'd'
    torch_dtype = torch.float64