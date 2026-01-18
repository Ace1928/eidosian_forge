import contextlib
from collections.abc import Iterable
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Tuple, Type, Union
from torch.utils.data.dataloader import _BaseDataLoaderIter, _MultiProcessingDataLoaderIter
from typing_extensions import Self, TypedDict, override
from lightning_fabric.utilities.data import sized_len
from lightning_fabric.utilities.types import _Stateful
from pytorch_lightning.utilities._pytree import _map_and_unflatten, _tree_flatten, tree_unflatten
def _load_current_iterator(self) -> None:
    if self._iterator_idx < len(self.iterables):
        self.iterators = [iter(self.iterables[self._iterator_idx])]
    else:
        self.iterators = []