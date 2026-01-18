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
def _read_axes(self, where, start: int | None=None, stop: int | None=None) -> list[tuple[np.ndarray, np.ndarray] | tuple[Index, Index]]:
    """
        Create the axes sniffed from the table.

        Parameters
        ----------
        where : ???
        start : int or None, default None
        stop : int or None, default None

        Returns
        -------
        List[Tuple[index_values, column_values]]
        """
    selection = Selection(self, where=where, start=start, stop=stop)
    values = selection.select()
    results = []
    for a in self.axes:
        a.set_info(self.info)
        res = a.convert(values, nan_rep=self.nan_rep, encoding=self.encoding, errors=self.errors)
        results.append(res)
    return results