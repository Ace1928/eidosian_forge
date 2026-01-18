from __future__ import annotations
from collections.abc import (
import datetime
from functools import partial
from io import BytesIO
import os
from textwrap import fill
from typing import (
import warnings
import zipfile
from pandas._config import config
from pandas._libs import lib
from pandas._libs.parsers import STR_NA_VALUES
from pandas.compat._optional import (
from pandas.errors import EmptyDataError
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend
from pandas.core.dtypes.common import (
from pandas.core.frame import DataFrame
from pandas.core.shared_docs import _shared_docs
from pandas.util.version import Version
from pandas.io.common import (
from pandas.io.excel._util import (
from pandas.io.parsers import TextParser
from pandas.io.parsers.readers import validate_integer
def _calc_rows(self, header: int | Sequence[int] | None, index_col: int | Sequence[int] | None, skiprows: Sequence[int] | int | Callable[[int], object] | None, nrows: int | None) -> int | None:
    """
        If nrows specified, find the number of rows needed from the
        file, otherwise return None.


        Parameters
        ----------
        header : int, list of int, or None
            See read_excel docstring.
        index_col : int, str, list of int, or None
            See read_excel docstring.
        skiprows : list-like, int, callable, or None
            See read_excel docstring.
        nrows : int or None
            See read_excel docstring.

        Returns
        -------
        int or None
        """
    if nrows is None:
        return None
    if header is None:
        header_rows = 1
    elif is_integer(header):
        header = cast(int, header)
        header_rows = 1 + header
    else:
        header = cast(Sequence, header)
        header_rows = 1 + header[-1]
    if is_list_like(header) and index_col is not None:
        header = cast(Sequence, header)
        if len(header) > 1:
            header_rows += 1
    if skiprows is None:
        return header_rows + nrows
    if is_integer(skiprows):
        skiprows = cast(int, skiprows)
        return header_rows + nrows + skiprows
    if is_list_like(skiprows):

        def f(skiprows: Sequence, x: int) -> bool:
            return x in skiprows
        skiprows = cast(Sequence, skiprows)
        return self._check_skiprows_func(partial(f, skiprows), header_rows + nrows)
    if callable(skiprows):
        return self._check_skiprows_func(skiprows, header_rows + nrows)
    return None