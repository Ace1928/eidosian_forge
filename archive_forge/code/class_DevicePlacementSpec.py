from abc import ABC, abstractmethod
from dataclasses import dataclass
import functools
from typing import Callable, Dict, List, TYPE_CHECKING
import torch
from ._internals import (
from torch.distributed._shard.metadata import ShardMetadata
import torch.distributed._shard.sharded_tensor.metadata as sharded_tensor_meta
from torch.distributed._shard.op_registry_utils import _decorator_func
@dataclass
class DevicePlacementSpec(PlacementSpec):
    """
    Associates placement of an entity with a single device.

    Args:
        device(:class:`torch.distributed._remote_device`): The device to place the entity on.
    """
    device: torch.distributed._remote_device

    def __post_init__(self):
        if not isinstance(self.device, torch.distributed._remote_device):
            self.device = torch.distributed._remote_device(self.device)