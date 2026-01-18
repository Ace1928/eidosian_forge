from __future__ import annotations
from collections import abc
from datetime import datetime
import functools
from itertools import zip_longest
import operator
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.internals import BlockValuesRefs
import pandas._libs.join as libjoin
from pandas._libs.lib import (
from pandas._libs.tslibs import (
from pandas._typing import (
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import (
from pandas.core.dtypes.astype import (
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import is_dict_like
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.accessor import CachedAccessor
import pandas.core.algorithms as algos
from pandas.core.array_algos.putmask import (
from pandas.core.arrays import (
from pandas.core.arrays.string_ import (
from pandas.core.base import (
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.indexers import (
from pandas.core.indexes.frozen import FrozenList
from pandas.core.missing import clean_reindex_fill_method
from pandas.core.ops import get_op_result_name
from pandas.core.ops.invalid import make_invalid_op
from pandas.core.sorting import (
from pandas.core.strings.accessor import StringMethods
from pandas.io.formats.printing import (
def _get_default_index_names(self, names: Hashable | Sequence[Hashable] | None=None, default=None) -> list[Hashable]:
    """
        Get names of index.

        Parameters
        ----------
        names : int, str or 1-dimensional list, default None
            Index names to set.
        default : str
            Default name of index.

        Raises
        ------
        TypeError
            if names not str or list-like
        """
    from pandas.core.indexes.multi import MultiIndex
    if names is not None:
        if isinstance(names, (int, str)):
            names = [names]
    if not isinstance(names, list) and names is not None:
        raise ValueError('Index names must be str or 1-dimensional list')
    if not names:
        if isinstance(self, MultiIndex):
            names = com.fill_missing_names(self.names)
        else:
            names = [default] if self.name is None else [self.name]
    return names