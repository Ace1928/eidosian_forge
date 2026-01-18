import abc
from dataclasses import dataclass
from typing import Any, List
from ray.data.block import Block, DataBatch
from ray.types import ObjectRef
@dataclass
class Batch:
    """A batch of data with a corresponding index.

    Attributes:
        batch_idx: The global index of this batch so that downstream operations can
            maintain ordering.
        data: The batch of data.
    """
    batch_idx: int
    data: DataBatch