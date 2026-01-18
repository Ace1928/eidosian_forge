from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar
from .. import logging
@dataclass
class StateDictSplit:
    is_sharded: bool = field(init=False)
    metadata: Dict[str, Any]
    filename_to_tensors: Dict[str, List[str]]
    tensor_to_filename: Dict[str, str]

    def __post_init__(self):
        self.is_sharded = len(self.filename_to_tensors) > 1