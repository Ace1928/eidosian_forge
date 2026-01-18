from __future__ import annotations
import contextlib
import functools
import inspect
import io
import itertools
import math
import os
import re
import sys
import warnings
from collections.abc import (
from enum import Enum
from pathlib import Path
from typing import (
import numpy as np
import pandas as pd
from xarray.namedarray.utils import (  # noqa: F401
class HiddenKeyDict(MutableMapping[K, V]):
    """Acts like a normal dictionary, but hides certain keys."""
    __slots__ = ('_data', '_hidden_keys')

    def __init__(self, data: MutableMapping[K, V], hidden_keys: Iterable[K]):
        self._data = data
        self._hidden_keys = frozenset(hidden_keys)

    def _raise_if_hidden(self, key: K) -> None:
        if key in self._hidden_keys:
            raise KeyError(f'Key `{key!r}` is hidden.')

    def __setitem__(self, key: K, value: V) -> None:
        self._raise_if_hidden(key)
        self._data[key] = value

    def __getitem__(self, key: K) -> V:
        self._raise_if_hidden(key)
        return self._data[key]

    def __delitem__(self, key: K) -> None:
        self._raise_if_hidden(key)
        del self._data[key]

    def __iter__(self) -> Iterator[K]:
        for k in self._data:
            if k not in self._hidden_keys:
                yield k

    def __len__(self) -> int:
        num_hidden = len(self._hidden_keys & self._data.keys())
        return len(self._data) - num_hidden