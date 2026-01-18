import builtins
import torch
from . import _dtypes_impl
class int32(signedinteger):
    name = 'int32'
    typecode = 'i'
    torch_dtype = torch.int32