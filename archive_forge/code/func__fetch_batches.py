import math
from contextlib import suppress
from typing import Callable, List, Optional, Union
import torch
from torch.utils.data import BatchSampler, DataLoader, IterableDataset, RandomSampler
from .logging import get_logger
from .state import AcceleratorState, DistributedType, GradientState, is_torch_xla_available
from .utils import (
def _fetch_batches(self, iterator):
    batches, batch = (None, None)
    if self.state.process_index == 0:
        try:
            if self.split_batches:
                batch = next(iterator)
            else:
                batches = []
                for _ in range(self.state.num_processes):
                    batches.append(next(iterator))
                try:
                    batch = concatenate(batches, dim=0)
                except RuntimeError as e:
                    raise RuntimeError("You can't use batches of different size with `dispatch_batches=True` or when using an `IterableDataset`.either pass `dispatch_batches=False` and have each process fetch its own batch  or pass `split_batches=True`. By doing so, the main process will fetch a full batch and slice it into `num_processes` batches for each process.") from e
            batch_info = [get_data_structure(batch), False]
        except StopIteration:
            batch_info = [None, True]
    else:
        batch_info = [None, self._stop_iteration]
    broadcast_object_list(batch_info)
    self._stop_iteration = batch_info[1]
    if self._stop_iteration:
        if not self.split_batches and (not self._drop_last):
            if self.state.process_index == 0 and len(batches) > 0:
                batch = concatenate(batches, dim=0)
                batch_info = [get_data_structure(batch), False]
            else:
                batch_info = [None, True]
            broadcast_object_list(batch_info)
    return (batch, batch_info)