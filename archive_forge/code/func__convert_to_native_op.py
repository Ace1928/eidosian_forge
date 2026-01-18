from abc import ABC, abstractmethod
from typing import Any, List, Optional
from torch import Tensor
from typing_extensions import Self
from lightning_fabric.utilities.types import CollectibleGroup
@classmethod
@abstractmethod
def _convert_to_native_op(cls, op: str) -> Any:
    ...