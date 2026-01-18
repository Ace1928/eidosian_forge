import builtins
import torch
from . import _dtypes_impl
class bool_(generic):
    name = 'bool_'
    typecode = '?'
    torch_dtype = torch.bool