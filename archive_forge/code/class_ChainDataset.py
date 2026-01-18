import bisect
import warnings
import math
from typing import (
from torch import default_generator, randperm
from torch._utils import _accumulate
from ... import Generator, Tensor
class ChainDataset(IterableDataset):
    """Dataset for chaining multiple :class:`IterableDataset` s.

    This class is useful to assemble different existing dataset streams. The
    chaining operation is done on-the-fly, so concatenating large-scale
    datasets with this class will be efficient.

    Args:
        datasets (iterable of IterableDataset): datasets to be chained together
    """

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__()
        self.datasets = datasets

    def __iter__(self):
        for d in self.datasets:
            assert isinstance(d, IterableDataset), 'ChainDataset only supports IterableDataset'
            yield from d

    def __len__(self):
        total = 0
        for d in self.datasets:
            assert isinstance(d, IterableDataset), 'ChainDataset only supports IterableDataset'
            total += len(d)
        return total