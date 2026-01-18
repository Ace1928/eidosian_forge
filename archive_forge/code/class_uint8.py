import builtins
import torch
from . import _dtypes_impl
class uint8(unsignedinteger):
    name = 'uint8'
    typecode = 'B'
    torch_dtype = torch.uint8