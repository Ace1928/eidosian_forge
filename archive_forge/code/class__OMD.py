import abc
import configparser as cp
import fnmatch
from functools import wraps
import inspect
from io import BufferedReader, IOBase
import logging
import os
import os.path as osp
import re
import sys
from git.compat import defenc, force_text
from git.util import LockFile
from typing import (
from git.types import Lit_config_levels, ConfigLevels_Tup, PathLike, assert_never, _T
class _OMD(OrderedDict_OMD):
    """Ordered multi-dict."""

    def __setitem__(self, key: str, value: _T) -> None:
        super().__setitem__(key, [value])

    def add(self, key: str, value: Any) -> None:
        if key not in self:
            super().__setitem__(key, [value])
            return
        super().__getitem__(key).append(value)

    def setall(self, key: str, values: List[_T]) -> None:
        super().__setitem__(key, values)

    def __getitem__(self, key: str) -> Any:
        return super().__getitem__(key)[-1]

    def getlast(self, key: str) -> Any:
        return super().__getitem__(key)[-1]

    def setlast(self, key: str, value: Any) -> None:
        if key not in self:
            super().__setitem__(key, [value])
            return
        prior = super().__getitem__(key)
        prior[-1] = value

    def get(self, key: str, default: Union[_T, None]=None) -> Union[_T, None]:
        return super().get(key, [default])[-1]

    def getall(self, key: str) -> List[_T]:
        return super().__getitem__(key)

    def items(self) -> List[Tuple[str, _T]]:
        """List of (key, last value for key)."""
        return [(k, self[k]) for k in self]

    def items_all(self) -> List[Tuple[str, List[_T]]]:
        """List of (key, list of values for key)."""
        return [(k, self.getall(k)) for k in self]