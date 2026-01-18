import warnings
from abc import ABC, abstractmethod
from collections import deque
import copy as copymodule
from typing import Any, Callable, Iterator, List, Literal, Optional, Sized, Tuple, TypeVar, Deque
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes._hook_iterator import _SnapshotState
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.utils.common import StreamWrapper, _check_unpickable_fn
class _ChildDataPipe(IterDataPipe):
    """
    Iterable Datapipe that is a child of a main DataPipe.

    The instance of this class will pass its instance_id to get the next value from its main DataPipe.

    Note:
        ChildDataPipe, like all other IterDataPipe, follows the single iterator per IterDataPipe constraint.
        Since ChildDataPipes share a common buffer, when an iterator is created for one of the ChildDataPipes,
        the previous iterators  for all ChildDataPipes must be invalidated, with the exception when a ChildDataPipe
        hasn't had an iterator created from it since the last invalidation. See the example below.

    Example:
        >>> # xdoctest: +REQUIRES(module:torchdata)
        >>> # Singler Iterator per IteraDataPipe Invalidation
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> source_dp = IterableWrapper(range(10))
        >>> cdp1, cdp2 = source_dp.fork(num_instances=2)
        >>> it1, it2 = iter(cdp1), iter(cdp2)
        >>> it3 = iter(cdp1)
        >>> # The line above invalidates `it1` and `it2`, and resets `ForkerIterDataPipe`.
        >>> it4 = iter(cdp2)
        >>> # The line above doesn't invalidate `it3`, because an iterator for `cdp2` hasn't been created since
        >>> # the last invalidation.

    Args:
        main_datapipe: Main DataPipe with a method 'get_next_element_by_instance(instance_id)'
        instance_id: integer identifier of this instance
    """
    _is_child_datapipe: bool = True

    def __init__(self, main_datapipe: IterDataPipe, instance_id: int):
        assert isinstance(main_datapipe, _ContainerTemplate)
        self.main_datapipe: IterDataPipe = main_datapipe
        self.instance_id = instance_id

    def __iter__(self):
        return self.main_datapipe.get_next_element_by_instance(self.instance_id)

    def __len__(self):
        return self.main_datapipe.get_length_by_instance(self.instance_id)

    def _set_main_datapipe_valid_iterator_id(self) -> int:
        """
        Update the valid iterator ID for both this DataPipe object and `main_datapipe`.

        `main_datapipe.reset()` is called when the ID is incremented to a new generation.
        """
        if self.main_datapipe._valid_iterator_id is None:
            self.main_datapipe._valid_iterator_id = 0
        elif self.main_datapipe._valid_iterator_id == self._valid_iterator_id:
            self.main_datapipe._valid_iterator_id += 1
            if not self.main_datapipe.is_every_instance_exhausted():
                warnings.warn('Some child DataPipes are not exhausted when __iter__ is called. We are resetting the buffer and each child DataPipe will read from the start again.', UserWarning)
            self.main_datapipe.reset()
        self._valid_iterator_id = self.main_datapipe._valid_iterator_id
        return self._valid_iterator_id

    def _check_valid_iterator_id(self, iterator_id) -> bool:
        """Check the valid iterator ID against that of DataPipe object and that of `main_datapipe`."""
        return iterator_id == self._valid_iterator_id and iterator_id == self.main_datapipe._valid_iterator_id