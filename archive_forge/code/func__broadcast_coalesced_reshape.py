import torch
from ..modules import Module
from . import comm
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Sequence, Set, TypeVar, Union, cast
from torch._utils import _get_device_index
from collections import OrderedDict
def _broadcast_coalesced_reshape(tensors: Sequence[torch.Tensor], devices: Sequence[Union[int, torch.device]], detach: bool=False) -> List[List[torch.Tensor]]:
    from ._functions import Broadcast
    if detach:
        return comm.broadcast_coalesced(tensors, devices)
    elif len(tensors) > 0:
        tensor_copies = Broadcast.apply(devices, *tensors)
        return [tensor_copies[i:i + len(tensors)] for i in range(0, len(tensor_copies), len(tensors))]
    else:
        return []