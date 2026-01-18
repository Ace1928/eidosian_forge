import functools
import inspect
import os
from collections import OrderedDict
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Dict, Generator, Iterable, Optional, Sized, Tuple, Type, Union
from lightning_utilities.core.inheritance import get_all_subclasses
from torch.utils.data import BatchSampler, DataLoader, IterableDataset, Sampler
from typing_extensions import TypeGuard
from lightning_fabric.utilities.enums import LightningEnum
from lightning_fabric.utilities.exceptions import MisconfigurationException
from lightning_fabric.utilities.rank_zero import rank_zero_warn
from lightning_fabric.utilities.seed import pl_worker_init_function
def _replace_value_in_saved_args(replace_key: str, replace_value: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any], default_kwargs: Dict[str, Any], arg_names: Tuple[str, ...]) -> Tuple[bool, Tuple[Any, ...], Dict[str, Any]]:
    """Tries to replace an argument value in a saved list of args and kwargs.

    Returns a tuple indicating success of the operation and modified saved args and kwargs

    """
    if replace_key in arg_names:
        replace_index = arg_names.index(replace_key)
        args = args[:replace_index] + (replace_value,) + args[replace_index + 1:]
        return (True, args, kwargs)
    if replace_key in kwargs or replace_key in default_kwargs:
        kwargs[replace_key] = replace_value
        return (True, args, kwargs)
    return (False, args, kwargs)