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
class ScheduleWrapper:
    """This class is used to override the schedule logic from the profiler and perform recording for both
    `training_step`, `validation_step`."""

    def __init__(self, schedule: Callable) -> None:
        if not _KINETO_AVAILABLE:
            raise ModuleNotFoundError('You are trying to use `ScheduleWrapper` which require kineto install.')
        self._schedule = schedule
        self.reset()

    def reset(self) -> None:
        self._num_training_step = 0
        self._num_validation_step = 0
        self._num_test_step = 0
        self._num_predict_step = 0
        self._training_step_reached_end = False
        self._validation_step_reached_end = False
        self._test_step_reached_end = False
        self._predict_step_reached_end = False
        self._current_action: Optional[str] = None
        self._prev_schedule_action: Optional[ProfilerAction] = None
        self._start_action_name: Optional[str] = None

    def setup(self, start_action_name: str) -> None:
        self._start_action_name = start_action_name

    def pre_step(self, current_action: str) -> None:
        self._current_action = current_action

    @property
    def is_training(self) -> bool:
        assert self._current_action is not None
        return self._current_action.endswith('training_step')

    @property
    def is_validating(self) -> bool:
        assert self._current_action is not None
        return self._current_action.endswith('validation_step')

    @property
    def is_testing(self) -> bool:
        assert self._current_action is not None
        return self._current_action.endswith('test_step')

    @property
    def is_predicting(self) -> bool:
        assert self._current_action is not None
        return self._current_action.endswith('predict_step')

    @property
    def num_step(self) -> int:
        if self.is_training:
            return self._num_training_step
        if self.is_validating:
            return self._num_validation_step
        if self.is_testing:
            return self._num_test_step
        if self.is_predicting:
            return self._num_predict_step
        return 0

    def _step(self) -> None:
        if self.is_training:
            self._num_training_step += 1
        elif self.is_validating:
            assert self._start_action_name is not None
            if self._start_action_name.endswith('on_fit_start'):
                if self._num_training_step > 0:
                    self._num_validation_step += 1
            else:
                self._num_validation_step += 1
        elif self.is_testing:
            self._num_test_step += 1
        elif self.is_predicting:
            self._num_predict_step += 1

    @property
    def has_finished(self) -> bool:
        if self.is_training:
            return self._training_step_reached_end
        if self.is_validating:
            return self._validation_step_reached_end
        if self.is_testing:
            return self._test_step_reached_end
        if self.is_predicting:
            return self._predict_step_reached_end
        return False

    def __call__(self, num_step: int) -> 'ProfilerAction':
        if self._current_action is None or self.has_finished:
            return ProfilerAction.NONE
        self._step()
        action = self._schedule(max(self.num_step, 0))
        if self._prev_schedule_action == ProfilerAction.RECORD and action == ProfilerAction.WARMUP:
            action = ProfilerAction.RECORD
        if action == ProfilerAction.RECORD_AND_SAVE:
            if self.is_training:
                self._training_step_reached_end = True
            elif self.is_validating:
                self._validation_step_reached_end = True
            elif self.is_testing:
                self._test_step_reached_end = True
            elif self.is_predicting:
                self._predict_step_reached_end = True
        self._prev_schedule_action = action
        return action