import builtins
import torch
from . import _dtypes_impl
class int64(signedinteger):
    name = 'int64'
    typecode = 'l'
    torch_dtype = torch.int64