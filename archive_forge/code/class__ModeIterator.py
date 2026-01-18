import contextlib
from collections.abc import Iterable
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Tuple, Type, Union
from torch.utils.data.dataloader import _BaseDataLoaderIter, _MultiProcessingDataLoaderIter
from typing_extensions import Self, TypedDict, override
from lightning_fabric.utilities.data import sized_len
from lightning_fabric.utilities.types import _Stateful
from pytorch_lightning.utilities._pytree import _map_and_unflatten, _tree_flatten, tree_unflatten
class _ModeIterator(Iterator[_ITERATOR_RETURN]):

    def __init__(self, iterables: List[Iterable], limits: Optional[List[Union[int, float]]]=None) -> None:
        if limits is not None and len(limits) != len(iterables):
            raise ValueError(f'Mismatch in number of limits ({len(limits)}) and number of iterables ({len(iterables)})')
        self.iterables = iterables
        self.iterators: List[Iterator] = []
        self._idx = 0
        self.limits = limits

    @override
    def __next__(self) -> _ITERATOR_RETURN:
        raise NotImplementedError

    @override
    def __iter__(self) -> Self:
        self.iterators = [iter(iterable) for iterable in self.iterables]
        self._idx = 0
        return self

    def __len__(self) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        self.iterators = []
        self._idx = 0

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state['iterators'] = [None if isinstance(iterator, _BaseDataLoaderIter) else iterator_state for iterator, iterator_state in zip(self.iterators, state['iterators'])]
        return state