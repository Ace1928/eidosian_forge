import dataclasses
import sys
import types
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Protocol, Union
from typing_extensions import TypeAlias
import torch
class DynamoCallbackFn(Protocol):

    def __call__(self, frame: DynamoFrameType, cache_entry: Optional[CacheEntry], frame_state: FrameState) -> Optional[GuardedCode]:
        ...