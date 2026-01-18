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
def _get_legacy_storage_class(self):
    if self.dtype not in _dtype_to_storage_type_map():
        return None
    storage_name = _dtype_to_storage_type_map()[self.dtype]
    if self.device.type not in ['cpu', 'cuda', torch._C._get_privateuse1_backend_name()]:
        return None
    module = torch if self.device.type == 'cpu' else getattr(torch, self.device.type)
    try:
        return getattr(module, storage_name)
    except AttributeError:
        return None