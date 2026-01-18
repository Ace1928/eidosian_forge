import ast
import importlib
import os
import pathlib
import sys
from glob import iglob
from configparser import ConfigParser
from importlib.machinery import ModuleSpec
from itertools import chain
from typing import (
from pathlib import Path
from types import ModuleType
from distutils.errors import DistutilsOptionError
from .._path import same_path as _same_path
from ..warnings import SetuptoolsWarning
class LazyMappingProxy(Mapping[_K, _V]):
    """Mapping proxy that delays resolving the target object, until really needed.

    >>> def obtain_mapping():
    ...     print("Running expensive function!")
    ...     return {"key": "value", "other key": "other value"}
    >>> mapping = LazyMappingProxy(obtain_mapping)
    >>> mapping["key"]
    Running expensive function!
    'value'
    >>> mapping["other key"]
    'other value'
    """

    def __init__(self, obtain_mapping_value: Callable[[], Mapping[_K, _V]]):
        self._obtain = obtain_mapping_value
        self._value: Optional[Mapping[_K, _V]] = None

    def _target(self) -> Mapping[_K, _V]:
        if self._value is None:
            self._value = self._obtain()
        return self._value

    def __getitem__(self, key: _K) -> _V:
        return self._target()[key]

    def __len__(self) -> int:
        return len(self._target())

    def __iter__(self) -> Iterator[_K]:
        return iter(self._target())