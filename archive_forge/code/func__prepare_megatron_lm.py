from __future__ import annotations
import contextlib
import functools
import json
import math
import os
import re
import shutil
import sys
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from functools import partial
from types import MethodType
from typing import Any, Callable, Union
import torch
import torch.utils.hooks as hooks
from .checkpointing import load_accelerator_state, load_custom_state, save_accelerator_state, save_custom_state
from .data_loader import DataLoaderDispatcher, prepare_data_loader, skip_first_batches
from .hooks import AlignDevicesHook
from .logging import get_logger
from .optimizer import AcceleratedOptimizer
from .scheduler import AcceleratedScheduler
from .state import AcceleratorState, GradientState, PartialState
from .tracking import LOGGER_TYPE_TO_CLASS, GeneralTracker, filter_trackers
from .utils import (
from .utils.constants import FSDP_PYTORCH_VERSION
from .utils.modeling import get_state_dict_offloaded_model
from .utils.other import is_compiled_module
from torch.distributed.algorithms.join import Join
def _prepare_megatron_lm(self, *args):
    megatron_lm_plugin = self.state.megatron_lm_plugin
    if not megatron_lm_plugin.megatron_dataset_flag:
        batch_sizes = [obj.batch_size for obj in args if hasattr(obj, 'batch_size')]
        if len(batch_sizes) == 0:
            raise ValueError('You must specify a training or evaluation dataloader in `accelerate.prepare()` when using Megatron-LM.')
        micro_batch_size = min(batch_sizes) if megatron_lm_plugin.is_train_batch_min else max(batch_sizes)
        if len(batch_sizes) > 1:
            logger.info(f'Since you passed both train and evaluation dataloader, `is_train_batch_min` (here {megatron_lm_plugin.is_train_batch_min} will decide the `train_batch_size` ({micro_batch_size}).')
    else:
        for obj in args:
            if isinstance(obj, MegatronLMDummyDataLoader):
                micro_batch_size = obj.dataset_args['micro_batch_size']
                break
    dp_degree = self.num_processes // (megatron_lm_plugin.tp_degree * megatron_lm_plugin.pp_degree)
    megatron_lm_plugin.set_training_args(micro_batch_size, dp_degree)
    model = None
    optimizer = None
    scheduler = None
    is_dummy_scheduler = False
    batch_data = None
    for obj in args:
        if isinstance(obj, torch.utils.data.DataLoader) and batch_data is None:
            batch_data = next(iter(obj))
        if isinstance(obj, torch.nn.Module):
            model = obj
        elif isinstance(obj, torch.optim.Optimizer):
            optimizer = obj
        elif isinstance(obj, (LRScheduler, MegatronLMDummyScheduler)):
            scheduler = obj
    if model is not None:
        megatron_lm_plugin.set_network_size_args(model, batch_data)
    if optimizer is not None:
        megatron_lm_plugin.set_optimizer_type(optimizer)
    if scheduler is not None:
        is_dummy_scheduler = isinstance(scheduler, MegatronLMDummyScheduler)
        if not is_dummy_scheduler:
            raise ValueError("You can't use a custom scheduler with Megatron-LM. Please use the `accelerate.utils.MegatronLMDummyScheduler` instead.")
        megatron_lm_plugin.set_scheduler_args(scheduler)
    megatron_lm_initialize(self, args_defaults=megatron_lm_plugin.megatron_lm_default_args)
    counter = 0
    result = []
    for obj in args:
        if isinstance(obj, torch.utils.data.DataLoader):
            result.append(megatron_lm_prepare_data_loader(self, obj))
            counter += 1
        elif isinstance(obj, MegatronLMDummyDataLoader):
            if counter == 0:
                obj.set_megatron_data_args()
                dataloaders = megatron_lm_prepare_data_loader(self, obj)
            result.append(dataloaders[counter])
            counter += 1
        else:
            result.append(obj)
    if model is not None:
        model = megatron_lm_prepare_model(self)
    if optimizer is not None:
        optimizer = megatron_lm_prepare_optimizer(self, model)
    if scheduler is not None:
        scheduler = megatron_lm_prepare_scheduler(self, optimizer, scheduler)
    if model is not None:
        model = MegatronEngine(self, model, optimizer, scheduler)
    if optimizer is not None:
        optimizer = MegatronLMOptimizerWrapper(optimizer)
    if scheduler is not None:
        scheduler = MegatronLMSchedulerWrapper(scheduler, optimizer)
    for i in range(len(result)):
        if isinstance(result[i], torch.nn.Module):
            result[i] = model
        elif isinstance(result[i], torch.optim.Optimizer):
            result[i] = optimizer
        elif isinstance(result[i], MegatronLMDummyScheduler):
            result[i] = scheduler
    if model is not None:
        self._models.append(model)
    if optimizer is not None:
        self._optimizers.append(optimizer)
    if scheduler is not None:
        self._schedulers.append(scheduler)
    if len(self._models) > 1:
        raise AssertionError("You can't use same `Accelerator()` instance with multiple models when using Megatron-LM")
    return tuple(result)