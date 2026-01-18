from __future__ import annotations
from collections.abc import (
import itertools
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.internals import (
from pandas._libs.tslibs import Timestamp
from pandas.errors import PerformanceWarning
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import infer_dtype_from_scalar
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
import pandas.core.algorithms as algos
from pandas.core.arrays import (
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.construction import (
from pandas.core.indexers import maybe_convert_indices
from pandas.core.indexes.api import (
from pandas.core.internals.base import (
from pandas.core.internals.blocks import (
from pandas.core.internals.ops import (
def _make_na_block(self, placement: BlockPlacement, fill_value=None, use_na_proxy: bool=False) -> Block:
    if use_na_proxy:
        assert fill_value is None
        shape = (len(placement), self.shape[1])
        vals = np.empty(shape, dtype=np.void)
        nb = NumpyBlock(vals, placement, ndim=2)
        return nb
    if fill_value is None:
        fill_value = np.nan
    shape = (len(placement), self.shape[1])
    dtype, fill_value = infer_dtype_from_scalar(fill_value)
    block_values = make_na_array(dtype, shape, fill_value)
    return new_block_2d(block_values, placement=placement)