import warnings
from abc import ABC, abstractmethod
from collections import deque
import copy as copymodule
from typing import Any, Callable, Iterator, List, Literal, Optional, Sized, Tuple, TypeVar, Deque
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes._hook_iterator import _SnapshotState
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.utils.common import StreamWrapper, _check_unpickable_fn
@functional_datapipe('fork')
class ForkerIterDataPipe(IterDataPipe):
    """
    Creates multiple instances of the same Iterable DataPipe (functional name: ``fork``).

    Args:
        datapipe: Iterable DataPipe being copied
        num_instances: number of instances of the datapipe to create
        buffer_size: this restricts how far ahead the leading child DataPipe
           can read relative to the slowest child DataPipe.
           Defaults to ``1000``. Use ``-1`` for the unlimited buffer.
        copy: copy strategy to use for items yielded by each branch. Supported
            options are ``None`` for no copying, ``"shallow"`` for shallow object
            copies, and ``"deep"`` for deep object copies. Defaults to ``None``.

    Note:
        All branches of the forked pipeline return the identical object unless
        the copy parameter is supplied. If the object is mutable or contains
        mutable objects, changing them in one branch will affect all others.

    Example:
        >>> # xdoctest: +REQUIRES(module:torchdata)
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> source_dp = IterableWrapper(range(5))
        >>> dp1, dp2 = source_dp.fork(num_instances=2)
        >>> list(dp1)
        [0, 1, 2, 3, 4]
        >>> list(dp2)
        [0, 1, 2, 3, 4]
    """

    def __new__(cls, datapipe: IterDataPipe, num_instances: int, buffer_size: int=1000, copy: Optional[Literal['shallow', 'deep']]=None):
        if num_instances < 1:
            raise ValueError(f'Expected `num_instances` larger than 0, but {num_instances} is found')
        if num_instances == 1:
            return datapipe
        container = _ForkerIterDataPipe(datapipe, num_instances, buffer_size, copy)
        return [_ChildDataPipe(container, i) for i in range(num_instances)]