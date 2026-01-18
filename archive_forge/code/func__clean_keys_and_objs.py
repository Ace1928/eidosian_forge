from __future__ import annotations
from collections import abc
from typing import (
import warnings
import numpy as np
from pandas._config import using_copy_on_write
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import isna
from pandas.core.arrays.categorical import (
import pandas.core.common as com
from pandas.core.indexes.api import (
from pandas.core.internals import concatenate_managers
def _clean_keys_and_objs(self, objs: Iterable[Series | DataFrame] | Mapping[HashableT, Series | DataFrame], keys) -> tuple[list[Series | DataFrame], Index | None]:
    if isinstance(objs, abc.Mapping):
        if keys is None:
            keys = list(objs.keys())
        objs_list = [objs[k] for k in keys]
    else:
        objs_list = list(objs)
    if len(objs_list) == 0:
        raise ValueError('No objects to concatenate')
    if keys is None:
        objs_list = list(com.not_none(*objs_list))
    else:
        clean_keys = []
        clean_objs = []
        if is_iterator(keys):
            keys = list(keys)
        if len(keys) != len(objs_list):
            warnings.warn('The behavior of pd.concat with len(keys) != len(objs) is deprecated. In a future version this will raise instead of truncating to the smaller of the two sequences', FutureWarning, stacklevel=find_stack_level())
        for k, v in zip(keys, objs_list):
            if v is None:
                continue
            clean_keys.append(k)
            clean_objs.append(v)
        objs_list = clean_objs
        if isinstance(keys, MultiIndex):
            keys = type(keys).from_tuples(clean_keys, names=keys.names)
        else:
            name = getattr(keys, 'name', None)
            keys = Index(clean_keys, name=name, dtype=getattr(keys, 'dtype', None))
    if len(objs_list) == 0:
        raise ValueError('All objects passed were None')
    return (objs_list, keys)