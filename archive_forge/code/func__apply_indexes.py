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
def _apply_indexes(indexes: Indexes[Index], args: Mapping[Any, Any], func: str) -> tuple[dict[Hashable, Index], dict[Hashable, Variable]]:
    new_indexes: dict[Hashable, Index] = {k: v for k, v in indexes.items()}
    new_index_variables: dict[Hashable, Variable] = {}
    for index, index_vars in indexes.group_by_index():
        index_dims = {d for var in index_vars.values() for d in var.dims}
        index_args = {k: v for k, v in args.items() if k in index_dims}
        if index_args:
            new_index = getattr(index, func)(index_args)
            if new_index is not None:
                new_indexes.update({k: new_index for k in index_vars})
                new_index_vars = new_index.create_variables(index_vars)
                new_index_variables.update(new_index_vars)
            else:
                for k in index_vars:
                    new_indexes.pop(k, None)
    return (new_indexes, new_index_variables)