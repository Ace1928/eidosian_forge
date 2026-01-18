from __future__ import annotations
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field, InitVar
from functools import lru_cache
import abc
import hashlib
import itertools, pathlib
import os
import pickle
import re
import textwrap
import typing as T
from . import coredata
from . import dependencies
from . import mlog
from . import programs
from .mesonlib import (
from .compilers import (
from .interpreterbase import FeatureNew, FeatureDeprecated
class ConfigurationData(HoldableObject):

    def __init__(self, initial_values: T.Optional[T.Union[T.Dict[str, T.Tuple[T.Union[str, int, bool], T.Optional[str]]], T.Dict[str, T.Union[str, int, bool]]]]=None):
        super().__init__()
        self.values: T.Dict[str, T.Tuple[T.Union[str, int, bool], T.Optional[str]]] = {k: v if isinstance(v, tuple) else (v, None) for k, v in initial_values.items()} if initial_values else {}
        self.used: bool = False

    def __repr__(self) -> str:
        return repr(self.values)

    def __contains__(self, value: str) -> bool:
        return value in self.values

    def __bool__(self) -> bool:
        return bool(self.values)

    def get(self, name: str) -> T.Tuple[T.Union[str, int, bool], T.Optional[str]]:
        return self.values[name]

    def keys(self) -> T.Iterator[str]:
        return self.values.keys()