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
def _init_kineto(self, profiler_kwargs: Any) -> None:
    has_schedule = 'schedule' in profiler_kwargs
    self._has_on_trace_ready = 'on_trace_ready' in profiler_kwargs
    schedule = profiler_kwargs.get('schedule', None)
    if schedule is not None:
        if not callable(schedule):
            raise MisconfigurationException(f'Schedule should be a callable. Found: {schedule}')
        action = schedule(0)
        if not isinstance(action, ProfilerAction):
            raise MisconfigurationException(f'Schedule should return a `torch.profiler.ProfilerAction`. Found: {action}')
    self._default_schedule()
    schedule = schedule if has_schedule else self._default_schedule()
    self._schedule = ScheduleWrapper(schedule) if schedule is not None else schedule
    self._profiler_kwargs['schedule'] = self._schedule
    activities = profiler_kwargs.get('activities', None)
    self._profiler_kwargs['activities'] = activities or self._default_activities()
    self._export_to_flame_graph = profiler_kwargs.get('export_to_flame_graph', False)
    self._metric = profiler_kwargs.get('metric', 'self_cpu_time_total')
    with_stack = profiler_kwargs.get('with_stack', False) or self._export_to_flame_graph
    self._profiler_kwargs['with_stack'] = with_stack