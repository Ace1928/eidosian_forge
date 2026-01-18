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
class FrozenMappingWarningOnValuesAccess(Frozen[K, V]):
    """
    Class which behaves like a Mapping but warns if the values are accessed.

    Temporary object to aid in deprecation cycle of `Dataset.dims` (see GH issue #8496).
    `Dataset.dims` is being changed from returning a mapping of dimension names to lengths to just
    returning a frozen set of dimension names (to increase consistency with `DataArray.dims`).
    This class retains backwards compatibility but raises a warning only if the return value
    of ds.dims is used like a dictionary (i.e. it doesn't raise a warning if used in a way that
    would also be valid for a FrozenSet, e.g. iteration).
    """
    __slots__ = ('mapping',)

    def _warn(self) -> None:
        emit_user_level_warning('The return type of `Dataset.dims` will be changed to return a set of dimension names in future, in order to be more consistent with `DataArray.dims`. To access a mapping from dimension names to lengths, please use `Dataset.sizes`.', FutureWarning)

    def __getitem__(self, key: K) -> V:
        self._warn()
        return super().__getitem__(key)

    @overload
    def get(self, key: K, /) -> V | None:
        ...

    @overload
    def get(self, key: K, /, default: V | T) -> V | T:
        ...

    def get(self, key: K, default: T | None=None) -> V | T | None:
        self._warn()
        return super().get(key, default)

    def keys(self) -> KeysView[K]:
        self._warn()
        return super().keys()

    def items(self) -> ItemsView[K, V]:
        self._warn()
        return super().items()

    def values(self) -> ValuesView[V]:
        self._warn()
        return super().values()