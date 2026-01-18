import inspect
import logging
import os
from functools import lru_cache, partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Dict, List, Optional, Type, Union
import torch
from torch import Tensor, nn
from torch.autograd.profiler import EventList, record_function
from torch.profiler import ProfilerAction, ProfilerActivity, tensorboard_trace_handler
from torch.utils.hooks import RemovableHandle
from typing_extensions import override
from lightning_fabric.accelerators.cuda import is_cuda_available
from pytorch_lightning.profilers.profiler import Profiler
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import WarningCache, rank_zero_warn
def _start_recording_forward(self, _: nn.Module, input: Tensor, record_name: str) -> Tensor:
    record = record_function('[pl][module]' + record_name)
    record.__enter__()
    self._records[record_name] = record
    return input