important optimization when chaining multiple CUDA graphs together, as it
from __future__ import annotations
import contextlib
import dataclasses
import functools
import gc
import itertools
import logging
import operator
import sys
import threading
import traceback
import warnings
import weakref
from collections import defaultdict
from enum import auto, Enum
from typing import (
import torch.fx
from torch import Tensor
from torch._dynamo.mutation_guard import GenerationTracker
from torch._dynamo.utils import preserve_rng_state
from torch._inductor.compile_fx import (
from torch.multiprocessing.reductions import StorageWeakRef
from torch.storage import UntypedStorage
from torch.types import _bool
from torch.utils import _pytree as pytree
from torch.utils.weak import TensorWeakRef
from . import config
class StorageWeakRefWrapper:
    """
    Wrapper around a storage weak ref. Will deallocate it upon expiration if invoked.
    """
    __slots__ = ['ref', '_data_ptr', 'extra_ref_check']
    storage_ref: Optional[StorageWeakRef]

    def __init__(self, inp: Union[Tensor, UntypedStorage], extra_ref_check: Optional[Callable[[], None]]=None):
        """
        extra_ref_check is an additional check we need to run to check if the
        weak ref has expired. in checking storage use count we assume extra_ref_check
        will hold an additional reference to the storage.
        """
        if isinstance(inp, Tensor):
            stor = inp.untyped_storage()
        else:
            assert isinstance(inp, UntypedStorage)
            stor = inp
        self.ref = StorageWeakRef(stor)
        self._data_ptr = stor.data_ptr()
        self.extra_ref_check = extra_ref_check

    @classmethod
    def from_weakref_and_data_ptr(cls, cdata, data_ptr, extra_ref_check=None):
        instance = cls.__new__(cls)
        instance._data_ptr = data_ptr
        instance.ref = StorageWeakRef.from_weakref(cdata)
        instance.extra_ref_check = extra_ref_check
        return instance

    def __call__(self) -> Optional[StorageWeakRefPointer]:
        if self.expired():
            return None
        return self.ref.cdata

    def swap_weakref(self, cdata):
        self.ref.__del__()
        self.ref.cdata = cdata

    def data_ptr(self) -> int:
        """NB: returns the data ptr even if the storage has expired"""
        return self._data_ptr

    def remove_extra_reference(self):
        self.extra_ref_check = None

    def expired(self):
        if self.extra_ref_check is not None and (not self.extra_ref_check()):
            return False
        stor_count = torch._C._storage_Use_Count(self.ref.cdata)
        return stor_count - (self.extra_ref_check is not None) == 0

    def __repr__(self):
        if self.ref is None or self.ref.expired():
            return f'StorageWeakRefWrapper to {self.data_ptr()}; dead'
        else:
            return f'StorageWeakRefWrapper to {self.data_ptr()}; alive'