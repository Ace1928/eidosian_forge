import abc
from dataclasses import dataclass
import io
from typing import List, Tuple, Any, Union, Optional
from enum import Enum, auto
import torch
from torch.distributed._shard.sharded_tensor.metadata import TensorProperties
from .metadata import (
@dataclass(frozen=True)
class ReadItem:
    type: LoadItemType
    dest_index: MetadataIndex
    dest_offsets: torch.Size
    storage_index: MetadataIndex
    storage_offsets: torch.Size
    lengths: torch.Size