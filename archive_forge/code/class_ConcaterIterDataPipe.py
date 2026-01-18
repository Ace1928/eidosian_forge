import warnings
from abc import ABC, abstractmethod
from collections import deque
import copy as copymodule
from typing import Any, Callable, Iterator, List, Literal, Optional, Sized, Tuple, TypeVar, Deque
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes._hook_iterator import _SnapshotState
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.utils.common import StreamWrapper, _check_unpickable_fn
@functional_datapipe('concat')
class ConcaterIterDataPipe(IterDataPipe):
    """
    Concatenates multiple Iterable DataPipes (functional name: ``concat``).

    The resulting DataPipe will yield all the elements from the first input DataPipe, before yielding from the subsequent ones.

    Args:
        datapipes: Iterable DataPipes being concatenated

    Example:
        >>> # xdoctest: +REQUIRES(module:torchdata)
        >>> import random
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp1 = IterableWrapper(range(3))
        >>> dp2 = IterableWrapper(range(5))
        >>> list(dp1.concat(dp2))
        [0, 1, 2, 0, 1, 2, 3, 4]
    """
    datapipes: Tuple[IterDataPipe]

    def __init__(self, *datapipes: IterDataPipe):
        if len(datapipes) == 0:
            raise ValueError('Expected at least one DataPipe, but got nothing')
        if not all((isinstance(dp, IterDataPipe) for dp in datapipes)):
            raise TypeError('Expected all inputs to be `IterDataPipe`')
        self.datapipes = datapipes

    def __iter__(self) -> Iterator:
        for dp in self.datapipes:
            yield from dp

    def __len__(self) -> int:
        if all((isinstance(dp, Sized) for dp in self.datapipes)):
            return sum((len(dp) for dp in self.datapipes))
        else:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")