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
def _handle_process_group_call(self, event: ProcessGroupTrackingEvent, *args: Sequence[Any]) -> None:
    torch.cuda.synchronize()
    if event == ProcessGroupTrackingEvent.allgather:
        outputs, inputs = args
        output_size = self._get_module_output_size(outputs)
        self._last_all_gather_memory += output_size
        if self._cumul_all_gather_memory:
            self._cumul_all_gather_memory[-1] += output_size