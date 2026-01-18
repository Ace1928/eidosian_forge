from __future__ import annotations
import inspect
import itertools
import warnings
from collections import defaultdict
from collections.abc import Iterable, Sequence
from contextlib import suppress
from copy import deepcopy
from dataclasses import field
from typing import TYPE_CHECKING, cast, overload
from warnings import warn
import numpy as np
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy
from ..exceptions import PlotnineError, PlotnineWarning
from ..mapping import aes
def _id_var(x: pd.Series[Any], drop: bool=False) -> list[int]:
    """
    Assign ids to items in x. If two items
    are the same, they get the same id.

    Parameters
    ----------
    x : array_like
        items to associate ids with
    drop : bool
        Whether to drop unused factor levels
    """
    if len(x) == 0:
        return []
    if array_kind.categorical(x):
        if drop:
            x = x.cat.remove_unused_categories()
            lst = list(x.cat.codes + 1)
        else:
            has_nan = any((np.isnan(i) for i in x if isinstance(i, float)))
            if has_nan:
                nan_code = -1
                new_nan_code = np.max(x.cat.codes) + 1
                lst = [val if val != nan_code else new_nan_code for val in x]
            else:
                lst = list(x.cat.codes + 1)
    else:
        try:
            levels = sorted(set(x))
        except TypeError:
            levels = multitype_sort(list(set(x)))
        lst = match(x, levels)
        lst = [item + 1 for item in lst]
    return lst