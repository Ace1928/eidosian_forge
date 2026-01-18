import contextlib
from collections.abc import Iterable
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Tuple, Type, Union
from torch.utils.data.dataloader import _BaseDataLoaderIter, _MultiProcessingDataLoaderIter
from typing_extensions import Self, TypedDict, override
from lightning_fabric.utilities.data import sized_len
from lightning_fabric.utilities.types import _Stateful
from pytorch_lightning.utilities._pytree import _map_and_unflatten, _tree_flatten, tree_unflatten
def _state_dicts(self) -> List[Dict[str, Any]]:
    """Returns the list of state dicts for iterables in `self.flattened` that are stateful."""
    return [loader.state_dict() for loader in self.flattened if isinstance(loader, _Stateful)]