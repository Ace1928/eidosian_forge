import contextlib
from collections.abc import Iterable
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Tuple, Type, Union
from torch.utils.data.dataloader import _BaseDataLoaderIter, _MultiProcessingDataLoaderIter
from typing_extensions import Self, TypedDict, override
from lightning_fabric.utilities.data import sized_len
from lightning_fabric.utilities.types import _Stateful
from pytorch_lightning.utilities._pytree import _map_and_unflatten, _tree_flatten, tree_unflatten
def _load_state_dicts(self, states: List[Dict[str, Any]]) -> None:
    """Loads the state dicts for iterables in `self.flattened` that are stateful."""
    if not states:
        return
    stateful_loaders = [loader for loader in self.flattened if isinstance(loader, _Stateful)]
    if len(stateful_loaders) != len(states):
        raise RuntimeError(f'The CombinedLoader has {len(stateful_loaders)} stateful loaders, but found {len(states)} states in the checkpoint. Please make sure you define the same dataloaders that were used when saving the checkpoint.')
    for loader, state_dict in zip(stateful_loaders, states):
        loader.load_state_dict(state_dict)