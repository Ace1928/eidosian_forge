import abc
from dataclasses import dataclass
import io
from typing import List, Tuple, Any, Union, Optional
from enum import Enum, auto
import torch
from torch.distributed._shard.sharded_tensor.metadata import TensorProperties
from .metadata import (
@dataclass(frozen=True)
class WriteItem:
    index: MetadataIndex
    type: WriteItemType
    tensor_data: Optional[TensorWriteData] = None