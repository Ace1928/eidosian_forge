import warnings
from abc import ABC, abstractmethod
from collections import deque
import copy as copymodule
from typing import Any, Callable, Iterator, List, Literal, Optional, Sized, Tuple, TypeVar, Deque
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes._hook_iterator import _SnapshotState
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.utils.common import StreamWrapper, _check_unpickable_fn
@functional_datapipe('demux')
class DemultiplexerIterDataPipe(IterDataPipe):
    """
    Splits the input DataPipe into multiple child DataPipes, using the given classification function (functional name: ``demux``).

    A list of the child DataPipes is returned from this operation.

    Args:
        datapipe: Iterable DataPipe being filtered
        num_instances: number of instances of the DataPipe to create
        classifier_fn: a function that maps values to an integer within the range ``[0, num_instances - 1]`` or ``None``
        drop_none: defaults to ``False``, if ``True``, the function will skip over elements classified as ``None``
        buffer_size: this defines the maximum number of inputs that the buffer can hold across all child
            DataPipes while waiting for their values to be yielded.
            Defaults to ``1000``. Use ``-1`` for the unlimited buffer.

    Examples:
        >>> # xdoctest: +REQUIRES(module:torchdata)
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> def odd_or_even(n):
        ...     return n % 2
        >>> source_dp = IterableWrapper(range(5))
        >>> dp1, dp2 = source_dp.demux(num_instances=2, classifier_fn=odd_or_even)
        >>> list(dp1)
        [0, 2, 4]
        >>> list(dp2)
        [1, 3]
        >>> # It can also filter out any element that gets `None` from the `classifier_fn`
        >>> def odd_or_even_no_zero(n):
        ...     return n % 2 if n != 0 else None
        >>> dp1, dp2 = source_dp.demux(num_instances=2, classifier_fn=odd_or_even_no_zero, drop_none=True)
        >>> list(dp1)
        [2, 4]
        >>> list(dp2)
        [1, 3]
    """

    def __new__(cls, datapipe: IterDataPipe, num_instances: int, classifier_fn: Callable[[T_co], Optional[int]], drop_none: bool=False, buffer_size: int=1000):
        if num_instances < 1:
            raise ValueError(f'Expected `num_instances` larger than 0, but {num_instances} is found')
        _check_unpickable_fn(classifier_fn)
        container = _DemultiplexerIterDataPipe(datapipe, num_instances, classifier_fn, drop_none, buffer_size)
        return [_ChildDataPipe(container, i) for i in range(num_instances)]