from dataclasses import dataclass, field
from typing import Dict, List, Union, Optional, Sequence, Any
from torch.distributed._shard.sharded_tensor.metadata import TensorProperties
from torch.distributed.checkpoint.stateful import StatefulT
import torch
from torch.distributed._shard.sharded_tensor import (
@dataclass(frozen=True)
class MetadataIndex:
    """This class represents a lookup key for items in a state dict or Metadata."""
    fqn: str
    'Fully Qualified Name of the object'
    offset: Optional[torch.Size] = None
    "If the object is a tensor, offset into the tensor we're looking for"
    index: Optional[int] = field(hash=False, compare=False, default=None)
    '\n    Index hint when searching for tensor chunk to speedup lookups (optional)\n\n    A common representation of a sharded tensor is as a list of chunks so to\n    find the index in such a list you need to linear search it.\n\n    When constructing an instance of MetadataIndex that points to that list,\n    one can provide the index as a hint and it will be probed first before\n    the linear search and thus making it significantly faster.\n    '

    def __init__(self, fqn: str, offset: Optional[Sequence[int]]=None, index: Optional[int]=None):
        object.__setattr__(self, 'fqn', fqn)
        object.__setattr__(self, 'index', index)
        if offset is not None:
            object.__setattr__(self, 'offset', torch.Size(offset))