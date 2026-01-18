from __future__ import annotations
import collections
from copy import deepcopy
import datetime as dt
from functools import partial
import gc
from json import loads
import operator
import pickle
import re
import sys
from typing import (
import warnings
import weakref
import numpy as np
from pandas._config import (
from pandas._libs import lib
from pandas._libs.lib import is_range_indexer
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
from pandas._typing import (
from pandas.compat import PYPY
from pandas.compat._constants import REF_COUNT
from pandas.compat._optional import import_optional_dependency
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import (
from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.array_algos.replace import should_use_regex
from pandas.core.arrays import ExtensionArray
from pandas.core.base import PandasObject
from pandas.core.construction import extract_array
from pandas.core.flags import Flags
from pandas.core.indexes.api import (
from pandas.core.internals import (
from pandas.core.internals.construction import (
from pandas.core.methods.describe import describe_ndframe
from pandas.core.missing import (
from pandas.core.reshape.concat import concat
from pandas.core.shared_docs import _shared_docs
from pandas.core.sorting import get_indexer_indexer
from pandas.core.window import (
from pandas.io.formats.format import (
from pandas.io.formats.printing import pprint_thing
@final
def _set_axis_name(self, name, axis: Axis=0, inplace: bool_t=False, copy: bool_t | None=True):
    """
        Set the name(s) of the axis.

        Parameters
        ----------
        name : str or list of str
            Name(s) to set.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            The axis to set the label. The value 0 or 'index' specifies index,
            and the value 1 or 'columns' specifies columns.
        inplace : bool, default False
            If `True`, do operation inplace and return None.
        copy:
            Whether to make a copy of the result.

        Returns
        -------
        Series, DataFrame, or None
            The same type as the caller or `None` if `inplace` is `True`.

        See Also
        --------
        DataFrame.rename : Alter the axis labels of :class:`DataFrame`.
        Series.rename : Alter the index labels or set the index name
            of :class:`Series`.
        Index.rename : Set the name of :class:`Index` or :class:`MultiIndex`.

        Examples
        --------
        >>> df = pd.DataFrame({"num_legs": [4, 4, 2]},
        ...                   ["dog", "cat", "monkey"])
        >>> df
                num_legs
        dog            4
        cat            4
        monkey         2
        >>> df._set_axis_name("animal")
                num_legs
        animal
        dog            4
        cat            4
        monkey         2
        >>> df.index = pd.MultiIndex.from_product(
        ...                [["mammal"], ['dog', 'cat', 'monkey']])
        >>> df._set_axis_name(["type", "name"])
                       num_legs
        type   name
        mammal dog        4
               cat        4
               monkey     2
        """
    axis = self._get_axis_number(axis)
    idx = self._get_axis(axis).set_names(name)
    inplace = validate_bool_kwarg(inplace, 'inplace')
    renamed = self if inplace else self.copy(deep=copy)
    if axis == 0:
        renamed.index = idx
    else:
        renamed.columns = idx
    if not inplace:
        return renamed