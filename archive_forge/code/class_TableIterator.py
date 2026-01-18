from __future__ import annotations
from contextlib import suppress
import copy
from datetime import (
import itertools
import os
import re
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.lib import is_string_array
from pandas._libs.tslibs import timezones
from pandas.compat._optional import import_optional_dependency
from pandas.compat.pickle_compat import patch_pickle
from pandas.errors import (
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import array_equivalent
from pandas import (
from pandas.core.arrays import (
import pandas.core.common as com
from pandas.core.computation.pytables import (
from pandas.core.construction import extract_array
from pandas.core.indexes.api import ensure_index
from pandas.core.internals import (
from pandas.io.common import stringify_path
from pandas.io.formats.printing import (
class TableIterator:
    """
    Define the iteration interface on a table

    Parameters
    ----------
    store : HDFStore
    s     : the referred storer
    func  : the function to execute the query
    where : the where of the query
    nrows : the rows to iterate on
    start : the passed start value (default is None)
    stop  : the passed stop value (default is None)
    iterator : bool, default False
        Whether to use the default iterator.
    chunksize : the passed chunking value (default is 100000)
    auto_close : bool, default False
        Whether to automatically close the store at the end of iteration.
    """
    chunksize: int | None
    store: HDFStore
    s: GenericFixed | Table

    def __init__(self, store: HDFStore, s: GenericFixed | Table, func, where, nrows, start=None, stop=None, iterator: bool=False, chunksize: int | None=None, auto_close: bool=False) -> None:
        self.store = store
        self.s = s
        self.func = func
        self.where = where
        if self.s.is_table:
            if nrows is None:
                nrows = 0
            if start is None:
                start = 0
            if stop is None:
                stop = nrows
            stop = min(nrows, stop)
        self.nrows = nrows
        self.start = start
        self.stop = stop
        self.coordinates = None
        if iterator or chunksize is not None:
            if chunksize is None:
                chunksize = 100000
            self.chunksize = int(chunksize)
        else:
            self.chunksize = None
        self.auto_close = auto_close

    def __iter__(self) -> Iterator:
        current = self.start
        if self.coordinates is None:
            raise ValueError('Cannot iterate until get_result is called.')
        while current < self.stop:
            stop = min(current + self.chunksize, self.stop)
            value = self.func(None, None, self.coordinates[current:stop])
            current = stop
            if value is None or not len(value):
                continue
            yield value
        self.close()

    def close(self) -> None:
        if self.auto_close:
            self.store.close()

    def get_result(self, coordinates: bool=False):
        if self.chunksize is not None:
            if not isinstance(self.s, Table):
                raise TypeError('can only use an iterator or chunksize on a table')
            self.coordinates = self.s.read_coordinates(where=self.where)
            return self
        if coordinates:
            if not isinstance(self.s, Table):
                raise TypeError('can only read_coordinates on a table')
            where = self.s.read_coordinates(where=self.where, start=self.start, stop=self.stop)
        else:
            where = self.where
        results = self.func(self.start, self.stop, where)
        self.close()
        return results