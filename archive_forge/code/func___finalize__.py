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
def __finalize__(self, other, method: str | None=None, **kwargs) -> Self:
    """
        Propagate metadata from other to self.

        Parameters
        ----------
        other : the object from which to get the attributes that we are going
            to propagate
        method : str, optional
            A passed method name providing context on where ``__finalize__``
            was called.

            .. warning::

               The value passed as `method` are not currently considered
               stable across pandas releases.
        """
    if isinstance(other, NDFrame):
        if other.attrs:
            self.attrs = deepcopy(other.attrs)
        self.flags.allows_duplicate_labels = other.flags.allows_duplicate_labels
        for name in set(self._metadata) & set(other._metadata):
            assert isinstance(name, str)
            object.__setattr__(self, name, getattr(other, name, None))
    if method == 'concat':
        if all((bool(obj.attrs) for obj in other.objs)):
            attrs = other.objs[0].attrs
            have_same_attrs = all((obj.attrs == attrs for obj in other.objs[1:]))
            if have_same_attrs:
                self.attrs = deepcopy(attrs)
        allows_duplicate_labels = all((x.flags.allows_duplicate_labels for x in other.objs))
        self.flags.allows_duplicate_labels = allows_duplicate_labels
    return self