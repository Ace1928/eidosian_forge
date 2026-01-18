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
def _wrap_attr_method(method: Callable, tag: _WrapAttrTag) -> Callable:
    """Wraps the ``__setattr__`` or ``__delattr__`` method of classes (currently :class:`~torch.utils.data.DataLoader`
    and :class:`~torch.utils.data.BatchSampler`) in order to enable re- instantiation of custom subclasses."""

    @functools.wraps(method)
    def wrapper(obj: Any, *args: Any) -> None:
        name, *_ = args
        prev_call_name, prev_call_method = getattr(obj, '__pl_current_call', (None, 'method'))
        first_call = not (prev_call_name == name and prev_call_method == tag)
        object.__setattr__(obj, '__pl_current_call', (name, tag))
        method(obj, *args)
        if first_call and (not getattr(obj, '__pl_inside_init', True)):
            attrs_record = getattr(obj, '__pl_attrs_record', [])
            attrs_record.append((args, tag))
            object.__setattr__(obj, '__pl_attrs_record', attrs_record)
        object.__setattr__(obj, '__pl_current_call', (prev_call_name, prev_call_method))
    return wrapper