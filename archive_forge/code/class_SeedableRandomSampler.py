import math
from contextlib import suppress
from typing import Callable, List, Optional, Union
import torch
from torch.utils.data import BatchSampler, DataLoader, IterableDataset, RandomSampler
from .logging import get_logger
from .state import AcceleratorState, DistributedType, GradientState, is_torch_xla_available
from .utils import (
class SeedableRandomSampler(RandomSampler):
    """
    Same as a random sampler, except that in `__iter__` a seed can be used.

    Needed specifically in distributed cases, when the random generator for each GPU needs to start from the same seed
    and be fully reproducable on multiple iterations.

    If a custom `generator` is passed, it will rely on its initial seed as well as the current iteration it is on
    (stored in `self.epoch`).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch = 0
        self.initial_seed = torch.random.initial_seed()

    def __iter__(self):
        if self.generator is None:
            self.generator = torch.Generator()
            self.generator.manual_seed(self.initial_seed)
        seed = self.epoch + self.initial_seed
        self.generator.manual_seed(seed)
        yield from super().__iter__()
        self.set_epoch(self.epoch + 1)

    def set_epoch(self, epoch: int):
        """Sets the current iteration of the sampler."""
        self.epoch = epoch