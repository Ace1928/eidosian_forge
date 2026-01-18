import contextlib
from collections.abc import Iterable
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Tuple, Type, Union
from torch.utils.data.dataloader import _BaseDataLoaderIter, _MultiProcessingDataLoaderIter
from typing_extensions import Self, TypedDict, override
from lightning_fabric.utilities.data import sized_len
from lightning_fabric.utilities.types import _Stateful
from pytorch_lightning.utilities._pytree import _map_and_unflatten, _tree_flatten, tree_unflatten
class _MaxSize(_ModeIterator):

    @override
    def __next__(self) -> _ITERATOR_RETURN:
        n = len(self.iterators)
        out = [None] * n
        all_exhausted = True
        for i in range(n):
            with contextlib.suppress(StopIteration):
                out[i] = next(self.iterators[i])
                all_exhausted = False
        if all_exhausted:
            raise StopIteration
        index = self._idx
        self._idx += 1
        return (out, index, 0)

    @override
    def __len__(self) -> int:
        lengths = _get_iterables_lengths(self.iterables)
        if self.limits is not None:
            return max((min(length, limit) for length, limit in zip(lengths, self.limits)))
        return max(lengths)