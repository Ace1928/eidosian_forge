import builtins
import torch
from . import _dtypes_impl
class float16(floating):
    name = 'float16'
    typecode = 'e'
    torch_dtype = torch.float16