import math
from contextlib import suppress
from typing import Callable, List, Optional, Union
import torch
from torch.utils.data import BatchSampler, DataLoader, IterableDataset, RandomSampler
from .logging import get_logger
from .state import AcceleratorState, DistributedType, GradientState, is_torch_xla_available
from .utils import (
class DataLoaderShard(DataLoader, DataLoaderStateMixin):
    """
    Subclass of a PyTorch `DataLoader` that will deal with device placement and current distributed setup.

    Args:
        dataset (`torch.utils.data.dataset.Dataset`):
            The dataset to use to build this datalaoder.
        device (`torch.device`, *optional*):
            If passed, the device to put all batches on.
        rng_types (list of `str` or [`~utils.RNGType`]):
            The list of random number generators to synchronize at the beginning of each iteration. Should be one or
            several of:

            - `"torch"`: the base torch random number generator
            - `"cuda"`: the CUDA random number generator (GPU only)
            - `"xla"`: the XLA random number generator (TPU only)
            - `"generator"`: an optional `torch.Generator`
        synchronized_generator (`torch.Generator`, *optional*):
            A random number generator to keep synchronized across processes.
        skip_batches (`int`, *optional*, defaults to 0):
            The number of batches to skip at the beginning.
        kwargs:
            All other keyword arguments to pass to the regular `DataLoader` initialization.

    **Available attributes:**

        - **total_batch_size** (`int`) -- Total batch size of the dataloader across all processes.
            Equal to the original batch size when `split_batches=True`; otherwise the original batch size * the total
            number of processes

        - **total_dataset_length** (`int`) -- Total length of the inner dataset across all processes.
    """

    def __init__(self, dataset, device=None, rng_types=None, synchronized_generator=None, skip_batches=0, _drop_last: bool=False, **kwargs):
        super().__init__(dataset, **kwargs)
        self.device = device
        self.rng_types = rng_types
        self.synchronized_generator = synchronized_generator
        self.skip_batches = skip_batches
        self.gradient_state = GradientState()
        self._drop_last = _drop_last
        self.iteration = 0

    def __iter__(self):
        if self.rng_types is not None:
            synchronize_rng_states(self.rng_types, self.synchronized_generator)
        self.begin()
        self.set_epoch(self.iteration)
        dataloader_iter = super().__iter__()
        try:
            current_batch = next(dataloader_iter)
        except StopIteration:
            yield
        batch_index = 0
        while True:
            try:
                if self.device is not None:
                    current_batch = send_to_device(current_batch, self.device)
                next_batch = next(dataloader_iter)
                if batch_index >= self.skip_batches:
                    yield current_batch
                batch_index += 1
                current_batch = next_batch
            except StopIteration:
                self.end_of_dataloader = True
                if batch_index >= self.skip_batches:
                    yield current_batch
                break
        self.iteration += 1
        self.end()

    def set_epoch(self, epoch: int):
        if self.iteration != epoch:
            self.iteration = epoch
        if hasattr(self.batch_sampler, 'sampler') and hasattr(self.batch_sampler.sampler, 'set_epoch'):
            self.batch_sampler.sampler.set_epoch(epoch)
        elif hasattr(self.dataset, 'set_epoch'):
            self.dataset.set_epoch(epoch)

    @property
    def total_batch_size(self):
        batch_sampler = self.sampler if isinstance(self.sampler, BatchSampler) else self.batch_sampler
        return batch_sampler.batch_size if getattr(batch_sampler, 'split_batches', False) else batch_sampler.batch_size * getattr(batch_sampler, 'num_processes', 1)

    @property
    def total_dataset_length(self):
        if hasattr(self.dataset, 'total_length'):
            return self.dataset.total_length
        else:
            return len(self.dataset)