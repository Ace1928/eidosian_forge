import builtins
import torch
from . import _dtypes_impl
class float32(floating):
    name = 'float32'
    typecode = 'f'
    torch_dtype = torch.float32