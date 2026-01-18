import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Set, Tuple
import torch.cuda.memory
import torch.cuda.nvtx
import torch.profiler
import torch.utils.hooks
from torch.utils._python_dispatch import TorchDispatchMode, _pop_mode_temporarily
from torch.utils._pytree import tree_map
from ..ops.common import FUNC_TO_XFORMERS_OPERATOR
from .device_limits import get_device_limits
from .profiler import _Profiler
@dataclass
class _OpInfo:
    flop_count: float = 0.0
    time_ms: float = 0.0
    io_bytes: int = 0
    is_exact_flop: bool = True
    op_name: str = ''
    op_suffix: str = ''
    stacktrace: Tuple[str, ...] = field(default_factory=tuple)
    ev_start: torch.cuda.Event = field(default_factory=lambda: torch.cuda.Event(enable_timing=True))
    ev_end: torch.cuda.Event = field(default_factory=lambda: torch.cuda.Event(enable_timing=True))
    hardware_tflops_limit: float = math.inf
    hardware_membw_limit: float = math.inf

    @property
    def time_membound_ms(self) -> float:
        assert self.time_ms > 0.0
        if self.io_bytes == 0:
            return 0.0
        return min(self.time_ms, 1000 * self.io_bytes / self.hardware_membw_limit)

    @property
    def time_computebound_ms(self) -> float:
        assert self.time_ms > 0.0
        tflop = self.flop_count / 1000 ** 4
        if tflop == 0.0:
            return 0.0
        return min(self.time_ms, 1000 * tflop / self.hardware_tflops_limit)

    def finalize(self) -> None:
        self.time_ms = self.ev_start.elapsed_time(self.ev_end)