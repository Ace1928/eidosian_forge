from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from functools import lru_cache
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Sequence, Set, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle
from fairscale.nn import FullyShardedDataParallel
class ProcessGroupTracker:
    """
    To be used as a wrapper around a ProcessGroup to track
    the calls to specific ProcessGroup function such as
    "allgather" calls.

    The tracker will send a notification to the listener
    when such calls occur.

    Best used in conjunction with LayerwiseMemoryTracker:

        ```
        # wrap the group used for FSDP
        group = ProcessGroupTracker(group)

        # use this group when creating FSDP blocks
        model = FullyShardedDataParallel(model, process_group=group),

        # monitor the model as before
        tracker = LayerwiseMemoryTracker()
        tracker.monitor(model)

        # the detailed traces will now contain information
        # about the amount of all gathered data
        tracker.memory_traces
        ```
    """

    def __init__(self, group: Any, listener: Optional[Callable]=None):
        self.group = group
        self.listener = listener

    def __getattr__(self, item: str) -> Any:
        if item == 'allgather':
            return self._build_wrapper(fct=self.group.allgather)
        elif item == '_allgather_base':
            return self._build_wrapper(fct=getattr(self.group, item))
        return getattr(self.group, item)

    def _build_wrapper(self, fct: Callable) -> Callable:

        def wrapper(output_tensors: Union[torch.Tensor, Sequence[torch.Tensor]], input_tensors: Union[torch.Tensor, Sequence[torch.Tensor]], *args: list, **kwargs: dict) -> Any:
            if self.listener is not None:
                self.listener(ProcessGroupTrackingEvent.allgather, output_tensors, input_tensors)
            return fct(output_tensors, input_tensors, *args, **kwargs)
        return wrapper