from __future__ import annotations
import functools
import operator
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Callable, Final, Generic, TypeVar, cast, overload
import numpy as np
import pandas as pd
from xarray.core import dtypes
from xarray.core.indexes import (
from xarray.core.types import T_Alignable
from xarray.core.utils import is_dict_like, is_full_slice
from xarray.core.variable import Variable, as_compatible_data, calculate_dimensions
def _reindex_one(self, obj: T_Alignable, matching_indexes: dict[MatchingIndexKey, Index]) -> T_Alignable:
    new_indexes, new_variables = self._get_indexes_and_vars(obj, matching_indexes)
    dim_pos_indexers = self._get_dim_pos_indexers(matching_indexes)
    return obj._reindex_callback(self, dim_pos_indexers, new_variables, new_indexes, self.fill_value, self.exclude_dims, self.exclude_vars)