from __future__ import annotations
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import lib
from pandas._libs.tslibs import OutOfBoundsDatetime
from pandas.errors import InvalidIndexError
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas.core import algorithms
from pandas.core.arrays import (
import pandas.core.common as com
from pandas.core.frame import DataFrame
from pandas.core.groupby import ops
from pandas.core.groupby.categorical import recode_for_groupby
from pandas.core.indexes.api import (
from pandas.core.series import Series
from pandas.io.formats.printing import pprint_thing
@cache_readonly
def _group_arraylike(self) -> ArrayLike:
    """
        Analogous to result_index, but holding an ArrayLike to ensure
        we can retain ExtensionDtypes.
        """
    if self._all_grouper is not None:
        return self._result_index._values
    elif self._passed_categorical:
        return self._group_index._values
    return self._codes_and_uniques[1]