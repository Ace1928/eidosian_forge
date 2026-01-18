import io
import torch
from ._utils import _type, _cuda, _hpu
from torch.types import Storage
from typing import cast, Any, Dict as _Dict, Optional as _Optional, TypeVar, Type, Union
import copy
import collections
from functools import lru_cache
import warnings
import threading
import functools
def _new_wrapped_storage(self, untyped_storage):
    assert type(untyped_storage) == torch.UntypedStorage
    if type(self) == TypedStorage:
        return TypedStorage(wrap_storage=untyped_storage, dtype=self.dtype, _internal=True)
    else:
        return type(self)(wrap_storage=untyped_storage)