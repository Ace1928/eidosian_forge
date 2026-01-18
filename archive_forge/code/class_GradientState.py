from __future__ import annotations
import logging
import math
import os
import threading
import warnings
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Optional
import torch
from .utils import (
from .utils.dataclasses import SageMakerDistributedType
class GradientState:
    """
    Singleton class that has information related to gradient synchronization for gradient accumulation

    **Available attributes:**

        - **end_of_dataloader** (`bool`) -- Whether we have reached the end the current dataloader
        - **remainder** (`int`) -- The number of extra samples that were added from padding the dataloader
        - **sync_gradients** (`bool`) -- Whether the gradients should be synced across all devices
        - **active_dataloader** (`Optional[DataLoader]`) -- The dataloader that is currently being iterated over
        - **dataloader_references** (`List[Optional[DataLoader]]`) -- A list of references to the dataloaders that are
            being iterated over
        - **num_steps** (`int`) -- The number of steps to accumulate over
        - **adjust_scheduler** (`bool`) -- Whether the scheduler should be adjusted to account for the gradient
            accumulation
        - **sync_with_dataloader** (`bool`) -- Whether the gradients should be synced at the end of the dataloader
            iteration and the number of total steps reset
        - **is_xla_gradients_synced** (`bool`) -- Whether the XLA gradients have been synchronized. It is initialized
          as false. Once gradients have been reduced before the optimizer step, this flag is set to true. Subsequently,
            after each step, the flag is reset to false. FSDP will always synchronize the gradients, hence
            is_xla_gradients_synced is always true.
    """
    _shared_state = SharedDict()

    def __init__(self, gradient_accumulation_plugin: Optional[GradientAccumulationPlugin]=None):
        self.__dict__ = self._shared_state
        if not self.initialized:
            self.sync_gradients = True
            self.active_dataloader = None
            self.dataloader_references = [None]
            self.plugin_kwargs = gradient_accumulation_plugin.to_kwargs() if gradient_accumulation_plugin is not None else {}
            self._is_xla_gradients_synced = False
        if gradient_accumulation_plugin is not None and self.plugin_kwargs != gradient_accumulation_plugin.to_kwargs():
            self.plugin_kwargs = gradient_accumulation_plugin.to_kwargs()

    @property
    def num_steps(self) -> int:
        """Returns the number of steps to accumulate over"""
        return self.plugin_kwargs.get('num_steps', 1)

    @property
    def adjust_scheduler(self) -> bool:
        """Returns whether the scheduler should be adjusted"""
        return self.plugin_kwargs.get('adjust_scheduler', False)

    @property
    def sync_with_dataloader(self) -> bool:
        """Returns whether the gradients should be synced at the end of the dataloader iteration and the number of total steps reset"""
        return self.plugin_kwargs.get('sync_with_dataloader', True)

    @property
    def initialized(self) -> bool:
        """Returns whether the `GradientState` has been initialized"""
        return GradientState._shared_state != {}

    @property
    def end_of_dataloader(self) -> bool:
        """Returns whether we have reached the end of the current dataloader"""
        if not self.in_dataloader:
            return False
        return self.active_dataloader.end_of_dataloader

    @property
    def remainder(self) -> int:
        """Returns the number of extra samples that were added from padding the dataloader"""
        if not self.in_dataloader:
            return -1
        return self.active_dataloader.remainder

    def __repr__(self):
        return f'Sync Gradients: {self.sync_gradients}\nAt end of current dataloader: {self.end_of_dataloader}\nExtra samples added: {self.remainder}\nGradient accumulation plugin: {self.plugin_kwargs}\n'

    @property
    def is_xla_gradients_synced(self):
        """Returns the value of is_xla_gradients_synced. FSDP will always synchronize the gradients, hence is_xla_gradients_synced is always true."""
        if parse_flag_from_env('ACCELERATE_USE_FSDP', default=False):
            return True
        return self._is_xla_gradients_synced

    @is_xla_gradients_synced.setter
    def is_xla_gradients_synced(self, is_synced):
        """Set the _is_xla_gradients_synced attribute."""
        self._is_xla_gradients_synced = is_synced

    def _set_sync_gradients(self, sync_gradients):
        """Private function that sets whether gradients should be synchronized. Users should not have to call this."""
        self.sync_gradients = sync_gradients
        if self.sync_gradients and is_torch_xla_available(check_is_tpu=True) and (PartialState().distributed_type == DistributedType.XLA):
            xm.mark_step()

    def _add_dataloader(self, dataloader):
        """Private function that adds a dataloader to `self.dataloader_references` and sets `in_dataloader` to `True`. Users should not have to call this."""
        self.active_dataloader = dataloader
        self.dataloader_references.append(self.active_dataloader)

    def _remove_dataloader(self, dataloader):
        """Private function that removes a dataloader from `self.dataloader_references` and sets `in_dataloader` to `False` if there are no more dataloaders. Users should not have to call this."""
        self.dataloader_references.remove(dataloader)
        self.active_dataloader = self.dataloader_references[-1]

    @property
    def in_dataloader(self) -> bool:
        """Returns whether the current process is in a dataloader"""
        return self.active_dataloader is not None

    @staticmethod
    def _reset_state():
        """Resets `_shared_state`, is used internally and should not be called"""
        GradientState._shared_state.clear()