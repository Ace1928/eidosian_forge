from __future__ import annotations  # type: ignore[attr-defined]
from dataclasses import dataclass
from typing import (
import copy
import warnings
from functools import reduce
import weakref
import threading
import torch
import torch.distributed as dist
from torch.distributed import rpc
from torch.distributed import distributed_c10d
from torch.distributed._shard.metadata import ShardMetadata
import torch.distributed._shard.sharding_spec as shard_spec
from torch.distributed._shard.sharding_spec.api import (
from torch.distributed._shard.sharding_spec._internals import (
from torch.distributed._shard._utils import (
from .metadata import TensorProperties, ShardedTensorMetadata
from .shard import Shard
from .reshard import reshuffle_local_shard, reshard_local_shard
from .utils import (
from torch.distributed.remote_device import _remote_device
from torch.utils import _pytree as pytree
def _create_tensor_from_params(*size, local_device, tensor_properties: TensorProperties):
    """ Helper to construct tensor from size, device and common params. """
    dtype = tensor_properties.dtype
    layout = tensor_properties.layout
    requires_grad = tensor_properties.requires_grad
    memory_format = tensor_properties.memory_format
    pin_memory = tensor_properties.pin_memory
    return torch.empty(*size, dtype=dtype, layout=layout, device=local_device, requires_grad=requires_grad, memory_format=memory_format, pin_memory=pin_memory)