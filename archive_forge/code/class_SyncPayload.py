from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, cast, Generic, List, Optional, Tuple, TypeVar, Union
import torch.distributed as dist
@dataclass
class SyncPayload(Generic[T]):
    stage_name: Optional[str]
    success: bool
    payload: T
    exception: Optional[Exception] = None