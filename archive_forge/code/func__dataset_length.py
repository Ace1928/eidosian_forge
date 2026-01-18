import contextlib
from collections.abc import Iterable
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Tuple, Type, Union
from torch.utils.data.dataloader import _BaseDataLoaderIter, _MultiProcessingDataLoaderIter
from typing_extensions import Self, TypedDict, override
from lightning_fabric.utilities.data import sized_len
from lightning_fabric.utilities.types import _Stateful
from pytorch_lightning.utilities._pytree import _map_and_unflatten, _tree_flatten, tree_unflatten
def _dataset_length(self) -> int:
    """Compute the total length of the datasets according to the current mode."""
    datasets = [getattr(dl, 'dataset', None) for dl in self.flattened]
    lengths = [length for ds in datasets if (length := sized_len(ds)) is not None]
    if not lengths:
        raise NotImplementedError('All datasets are iterable-style datasets.')
    fn = _SUPPORTED_MODES[self._mode]['fn']
    return fn(lengths)