import torch
import torch._prims_common as utils
from torch._decomp import register_decomposition
from torch._prims_common import TensorLikeType
from torch._prims_common.wrappers import out_wrapper
from torch._refs import _broadcast_shapes
def _make_conversion_method(name: str, dtype: torch.dtype):

    def fn(self: TensorLikeType, memory_format: torch.memory_format=torch.preserve_format) -> TensorLikeType:
        return self.to(dtype, memory_format=memory_format)
    fn.__name__ = name
    return fn