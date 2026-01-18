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
def _check_setitem_copy(self, t: str='setting', force: bool_t=False):
    """

        Parameters
        ----------
        t : str, the type of setting error
        force : bool, default False
           If True, then force showing an error.

        validate if we are doing a setitem on a chained copy.

        It is technically possible to figure out that we are setting on
        a copy even WITH a multi-dtyped pandas object. In other words, some
        blocks may be views while other are not. Currently _is_view will ALWAYS
        return False for multi-blocks to avoid having to handle this case.

        df = DataFrame(np.arange(0,9), columns=['count'])
        df['group'] = 'b'

        # This technically need not raise SettingWithCopy if both are view
        # (which is not generally guaranteed but is usually True.  However,
        # this is in general not a good practice and we recommend using .loc.
        df.iloc[0:5]['group'] = 'a'

        """
    if using_copy_on_write() or warn_copy_on_write():
        return
    if not (force or self._is_copy):
        return
    value = config.get_option('mode.chained_assignment')
    if value is None:
        return
    if self._is_copy is not None and (not isinstance(self._is_copy, str)):
        r = self._is_copy()
        if not gc.get_referents(r) or (r is not None and r.shape == self.shape):
            self._is_copy = None
            return
    if isinstance(self._is_copy, str):
        t = self._is_copy
    elif t == 'referent':
        t = '\nA value is trying to be set on a copy of a slice from a DataFrame\n\nSee the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy'
    else:
        t = '\nA value is trying to be set on a copy of a slice from a DataFrame.\nTry using .loc[row_indexer,col_indexer] = value instead\n\nSee the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy'
    if value == 'raise':
        raise SettingWithCopyError(t)
    if value == 'warn':
        warnings.warn(t, SettingWithCopyWarning, stacklevel=find_stack_level())