from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch._ops import OpOverload
from torch.distributed._tensor.placement_types import DTensorSpec
from torch.distributed.device_mesh import DeviceMesh
def _rebuild_tensor_from_dtensor_meta(arg) -> object:
    """ "
    This is used to propagate tensor metadata, must be under fake mode
    """
    assert arg.tensor_meta is not None, 'DTensorSpec does not contain tensor_meta.'
    return torch.empty_strided(arg.tensor_meta.shape, arg.tensor_meta.stride, dtype=arg.tensor_meta.dtype)