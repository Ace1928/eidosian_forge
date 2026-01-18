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
def _wrap_init_method(init: Callable, store_explicit_arg: Optional[str]=None) -> Callable:
    """Wraps the ``__init__`` method of classes (currently :class:`~torch.utils.data.DataLoader` and
    :class:`~torch.utils.data.BatchSampler`) in order to enable re-instantiation of custom subclasses."""

    @functools.wraps(init)
    def wrapper(obj: Any, *args: Any, **kwargs: Any) -> None:
        old_inside_init = getattr(obj, '__pl_inside_init', False)
        object.__setattr__(obj, '__pl_inside_init', True)
        params = inspect.signature(init).parameters
        parameters_defaults = OrderedDict(((param.name, param.default) for param in params.values() if param.name != 'self' and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)))
        param_names = tuple(parameters_defaults)[:len(args)]
        default_kwargs = {name: value for name, value in parameters_defaults.items() if name not in kwargs and name not in param_names and (value != inspect.Parameter.empty)}
        if not hasattr(obj, '__pl_saved_args'):
            object.__setattr__(obj, '__pl_saved_args', args)
            object.__setattr__(obj, '__pl_saved_kwargs', kwargs)
            object.__setattr__(obj, '__pl_saved_arg_names', param_names)
            object.__setattr__(obj, '__pl_saved_default_kwargs', default_kwargs)
        if store_explicit_arg is not None:
            if store_explicit_arg in param_names:
                object.__setattr__(obj, f'__{store_explicit_arg}', args[param_names.index(store_explicit_arg)])
            elif store_explicit_arg in kwargs:
                object.__setattr__(obj, f'__{store_explicit_arg}', kwargs[store_explicit_arg])
        init(obj, *args, **kwargs)
        object.__setattr__(obj, '__pl_inside_init', old_inside_init)
    return wrapper