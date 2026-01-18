from __future__ import annotations
from functools import wraps
import inspect
import re
from typing import (
import warnings
import weakref
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.internals import (
from pandas._libs.missing import NA
from pandas._typing import (
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_bool_kwarg
from pandas.core.dtypes.astype import (
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core import missing
import pandas.core.algorithms as algos
from pandas.core.array_algos.putmask import (
from pandas.core.array_algos.quantile import quantile_compat
from pandas.core.array_algos.replace import (
from pandas.core.array_algos.transforms import shift
from pandas.core.arrays import (
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.computation import expressions
from pandas.core.construction import (
from pandas.core.indexers import check_setitem_lengths
from pandas.core.indexes.base import get_values_for_csv
@final
def _maybe_downcast(self, blocks: list[Block], downcast, using_cow: bool, caller: str) -> list[Block]:
    if downcast is False:
        return blocks
    if self.dtype == _dtype_obj:
        if caller == 'fillna' and get_option('future.no_silent_downcasting'):
            return blocks
        nbs = extend_blocks([blk.convert(using_cow=using_cow, copy=not using_cow) for blk in blocks])
        if caller == 'fillna':
            if len(nbs) != len(blocks) or not all((x.dtype == y.dtype for x, y in zip(nbs, blocks))):
                warnings.warn("Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`", FutureWarning, stacklevel=find_stack_level())
        return nbs
    elif downcast is None:
        return blocks
    elif caller == 'where' and get_option('future.no_silent_downcasting') is True:
        return blocks
    else:
        nbs = extend_blocks([b._downcast_2d(downcast, using_cow) for b in blocks])
    if caller == 'where':
        if len(blocks) != len(nbs) or any((left.dtype != right.dtype for left, right in zip(blocks, nbs))):
            warnings.warn("Downcasting behavior in Series and DataFrame methods 'where', 'mask', and 'clip' is deprecated. In a future version this will not infer object dtypes or cast all-round floats to integers. Instead call result.infer_objects(copy=False) for object inference, or cast round floats explicitly. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`", FutureWarning, stacklevel=find_stack_level())
    return nbs