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
@property
def _total_steps(self) -> Union[int, float]:
    assert self._schedule is not None
    assert self._lightning_module is not None
    trainer = self._lightning_module.trainer
    if self._schedule.is_training:
        return trainer.num_training_batches
    if self._schedule.is_validating:
        num_val_batches = sum(trainer.num_val_batches) if isinstance(trainer.num_val_batches, list) else trainer.num_val_batches
        num_sanity_val_batches = sum(trainer.num_sanity_val_batches) if isinstance(trainer.num_sanity_val_batches, list) else trainer.num_sanity_val_batches
        return num_val_batches + num_sanity_val_batches
    if self._schedule.is_testing:
        num_test_batches = sum(trainer.num_test_batches) if isinstance(trainer.num_test_batches, list) else trainer.num_test_batches
        return num_test_batches
    if self._schedule.is_predicting:
        return sum(trainer.num_predict_batches)
    raise NotImplementedError('Unsupported schedule')