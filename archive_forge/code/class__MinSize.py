import contextlib
from collections.abc import Iterable
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Tuple, Type, Union
from torch.utils.data.dataloader import _BaseDataLoaderIter, _MultiProcessingDataLoaderIter
from typing_extensions import Self, TypedDict, override
from lightning_fabric.utilities.data import sized_len
from lightning_fabric.utilities.types import _Stateful
from pytorch_lightning.utilities._pytree import _map_and_unflatten, _tree_flatten, tree_unflatten
class _MinSize(_ModeIterator):

    @override
    def __next__(self) -> _ITERATOR_RETURN:
        out = [next(it) for it in self.iterators]
        index = self._idx
        self._idx += 1
        return (out, index, 0)

    @override
    def __len__(self) -> int:
        lengths = _get_iterables_lengths(self.iterables)
        return min(lengths + self.limits) if self.limits is not None else min(lengths)