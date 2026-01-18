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
def _maybe_wrap_index(self, idx, is_stop=False):
    if idx is None:
        if is_stop:
            return self._size()
        else:
            return 0
    else:
        if type(idx) != int:
            raise TypeError(f"can't index a {type(self)} with {type(idx)}")
        if is_stop:
            if idx > self._size() or idx < -self._size():
                raise IndexError(f'index {idx} out of range for storage of size {self.size()}')
            if idx > 0:
                return idx
            else:
                return idx % self._size()
        else:
            if idx >= self._size() or idx < -self._size():
                raise IndexError(f'index {idx} out of range for storage of size {self.size()}')
            return idx % self._size()