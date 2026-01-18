from __future__ import annotations
import collections
from collections import abc
from collections.abc import (
import functools
from inspect import signature
from io import StringIO
import itertools
import operator
import sys
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from numpy import ma
from pandas._config import (
from pandas._config.config import _get_option
from pandas._libs import (
from pandas._libs.hashtable import duplicated
from pandas._libs.lib import is_range_indexer
from pandas.compat import PYPY
from pandas.compat._constants import REF_COUNT
from pandas.compat._optional import import_optional_dependency
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import (
from pandas.util._validators import (
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.accessor import CachedAccessor
from pandas.core.apply import reconstruct_and_relabel_result
from pandas.core.array_algos.take import take_2d_multi
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import (
from pandas.core.arrays.sparse import SparseFrameAccessor
from pandas.core.construction import (
from pandas.core.generic import (
from pandas.core.indexers import check_key_length
from pandas.core.indexes.api import (
from pandas.core.indexes.multi import (
from pandas.core.indexing import (
from pandas.core.internals import (
from pandas.core.internals.construction import (
from pandas.core.methods import selectn
from pandas.core.reshape.melt import melt
from pandas.core.series import Series
from pandas.core.shared_docs import _shared_docs
from pandas.core.sorting import (
from pandas.io.common import get_handle
from pandas.io.formats import (
from pandas.io.formats.info import (
import pandas.plotting
def _repr_fits_horizontal_(self) -> bool:
    """
        Check if full repr fits in horizontal boundaries imposed by the display
        options width and max_columns.
        """
    width, height = console.get_console_size()
    max_columns = get_option('display.max_columns')
    nb_columns = len(self.columns)
    if max_columns and nb_columns > max_columns or (width and nb_columns > width // 2):
        return False
    if width is None or not console.in_interactive_session():
        return True
    if get_option('display.width') is not None or console.in_ipython_frontend():
        max_rows = 1
    else:
        max_rows = get_option('display.max_rows')
    buf = StringIO()
    d = self
    if max_rows is not None:
        d = d.iloc[:min(max_rows, len(d))]
    else:
        return True
    d.to_string(buf=buf)
    value = buf.getvalue()
    repr_width = max((len(line) for line in value.split('\n')))
    return repr_width < width