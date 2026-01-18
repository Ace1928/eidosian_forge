from __future__ import annotations
from collections import defaultdict
from copy import copy
import csv
import datetime
from enum import Enum
import itertools
from typing import (
import warnings
import numpy as np
from pandas._libs import (
import pandas._libs.ops as libops
from pandas._libs.parsers import STR_NA_VALUES
from pandas._libs.tslibs import parsing
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.astype import astype_array
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import isna
from pandas import (
from pandas.core import algorithms
from pandas.core.arrays import (
from pandas.core.arrays.boolean import BooleanDtype
from pandas.core.indexes.api import (
from pandas.core.series import Series
from pandas.core.tools import datetimes as tools
from pandas.io.common import is_potential_multi_index
@final
def _extract_multi_indexer_columns(self, header, index_names: Sequence[Hashable] | None, passed_names: bool=False) -> tuple[Sequence[Hashable], Sequence[Hashable] | None, Sequence[Hashable] | None, bool]:
    """
        Extract and return the names, index_names, col_names if the column
        names are a MultiIndex.

        Parameters
        ----------
        header: list of lists
            The header rows
        index_names: list, optional
            The names of the future index
        passed_names: bool, default False
            A flag specifying if names where passed

        """
    if len(header) < 2:
        return (header[0], index_names, None, passed_names)
    ic = self.index_col
    if ic is None:
        ic = []
    if not isinstance(ic, (list, tuple, np.ndarray)):
        ic = [ic]
    sic = set(ic)
    index_names = header.pop(-1)
    index_names, _, _ = self._clean_index_names(index_names, self.index_col)
    field_count = len(header[0])
    if not all((len(header_iter) == field_count for header_iter in header[1:])):
        raise ParserError('Header rows must have an equal number of columns.')

    def extract(r):
        return tuple((r[i] for i in range(field_count) if i not in sic))
    columns = list(zip(*(extract(r) for r in header)))
    names = columns.copy()
    for single_ic in sorted(ic):
        names.insert(single_ic, single_ic)
    if len(ic):
        col_names = [r[ic[0]] if r[ic[0]] is not None and r[ic[0]] not in self.unnamed_cols else None for r in header]
    else:
        col_names = [None] * len(header)
    passed_names = True
    return (names, index_names, col_names, passed_names)