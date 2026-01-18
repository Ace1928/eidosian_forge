from __future__ import annotations
import datetime
import sys
from collections.abc import Collection, Hashable, Iterator, Mapping, Sequence
from typing import (
import numpy as np
import pandas as pd
class Alignable(Protocol):
    """Represents any Xarray type that supports alignment.

    It may be ``Dataset``, ``DataArray`` or ``Coordinates``. This protocol class
    is needed since those types do not all have a common base class.

    """

    @property
    def dims(self) -> Frozen[Hashable, int] | tuple[Hashable, ...]:
        ...

    @property
    def sizes(self) -> Mapping[Hashable, int]:
        ...

    @property
    def xindexes(self) -> Indexes[Index]:
        ...

    def _reindex_callback(self, aligner: Aligner, dim_pos_indexers: dict[Hashable, Any], variables: dict[Hashable, Variable], indexes: dict[Hashable, Index], fill_value: Any, exclude_dims: frozenset[Hashable], exclude_vars: frozenset[Hashable]) -> Self:
        ...

    def _overwrite_indexes(self, indexes: Mapping[Any, Index], variables: Mapping[Any, Variable] | None=None) -> Self:
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterator[Hashable]:
        ...

    def copy(self, deep: bool=False) -> Self:
        ...