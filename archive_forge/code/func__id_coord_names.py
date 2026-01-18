from __future__ import annotations
import collections.abc
import copy
from collections import defaultdict
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast
import numpy as np
import pandas as pd
from xarray.core import formatting, nputils, utils
from xarray.core.indexing import (
from xarray.core.utils import (
@property
def _id_coord_names(self) -> dict[int, tuple[Hashable, ...]]:
    if self.__id_coord_names is None:
        id_coord_names: Mapping[int, list[Hashable]] = defaultdict(list)
        for k, v in self._coord_name_id.items():
            id_coord_names[v].append(k)
        self.__id_coord_names = {k: tuple(v) for k, v in id_coord_names.items()}
    return self.__id_coord_names