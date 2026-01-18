from abc import ABC, abstractmethod
from dataclasses import dataclass
import functools
from typing import Callable, Dict, List, TYPE_CHECKING
import torch
from ._internals import (
from torch.distributed._shard.metadata import ShardMetadata
import torch.distributed._shard.sharded_tensor.metadata as sharded_tensor_meta
from torch.distributed._shard.op_registry_utils import _decorator_func
class PlacementSpec(ABC):
    """
    Base class representing the placement of an entity. Subclasses of this
    class can be used to specify customized placements which might not be
    covered by existing APIs.
    """
    pass