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
@contextmanager
def _replace_dunder_methods(base_cls: Type, store_explicit_arg: Optional[str]=None) -> Generator[None, None, None]:
    """This context manager is used to add support for re-instantiation of custom (subclasses) of `base_cls`.

    It patches the ``__init__``, ``__setattr__`` and ``__delattr__`` methods.

    """
    classes = get_all_subclasses(base_cls) | {base_cls}
    for cls in classes:
        if '__init__' in cls.__dict__:
            cls.__old__init__ = cls.__init__
            cls.__init__ = _wrap_init_method(cls.__init__, store_explicit_arg)
        for patch_fn_name, tag in (('__setattr__', _WrapAttrTag.SET), ('__delattr__', _WrapAttrTag.DEL)):
            if patch_fn_name in cls.__dict__ or cls is base_cls:
                saved_name = f'__old{patch_fn_name}'
                setattr(cls, saved_name, getattr(cls, patch_fn_name))
                setattr(cls, patch_fn_name, _wrap_attr_method(getattr(cls, patch_fn_name), tag))
    yield
    for cls in classes:
        for patched_name in ('__setattr__', '__delattr__', '__init__'):
            if f'__old{patched_name}' in cls.__dict__:
                setattr(cls, patched_name, getattr(cls, f'__old{patched_name}'))
                delattr(cls, f'__old{patched_name}')