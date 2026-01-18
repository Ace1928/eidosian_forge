from __future__ import annotations
from collections import defaultdict
from typing import TYPE_CHECKING
import warnings
import numpy as np
from pandas._libs import (
from pandas.compat._optional import import_optional_dependency
from pandas.errors import DtypeWarning
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.concat import (
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas.core.indexes.api import ensure_index_from_sequences
from pandas.io.common import (
from pandas.io.parsers.base_parser import (
def _set_noconvert_columns(self) -> None:
    """
        Set the columns that should not undergo dtype conversions.

        Currently, any column that is involved with date parsing will not
        undergo such conversions.
        """
    assert self.orig_names is not None
    names_dict = {x: i for i, x in enumerate(self.orig_names)}
    col_indices = [names_dict[x] for x in self.names]
    noconvert_columns = self._set_noconvert_dtype_columns(col_indices, self.names)
    for col in noconvert_columns:
        self._reader.set_noconvert(col)