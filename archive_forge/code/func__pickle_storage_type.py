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
def _pickle_storage_type(self):
    try:
        return _dtype_to_storage_type_map()[self.dtype]
    except KeyError as e:
        raise KeyError(f'dtype {self.dtype} is not recognized') from e