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
@lru_cache(maxsize=None)
def _storage_type_to_dtype_map():
    dtype_map = {val: key for key, val in _dtype_to_storage_type_map().items()}
    return dtype_map