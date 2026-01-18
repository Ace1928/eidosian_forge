import functools
import torch
import torch.distributed as dist
from enum import Enum
class DQuantType(Enum):
    """
    Different quantization methods for auto_quantize API are identified here.

    auto_quantize API currently supports fp16 and bfp16 methods.
    """
    FP16 = ('fp16',)
    BFP16 = 'bfp16'

    def __str__(self) -> str:
        return self.value