from __future__ import annotations
from collections.abc import (
import datetime
from functools import partial
from typing import (
import uuid
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.lib import is_range_indexer
from pandas._typing import (
from pandas.errors import MergeError
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas import (
import pandas.core.algorithms as algos
from pandas.core.arrays import (
from pandas.core.arrays.string_ import StringDtype
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.frame import _merge_doc
from pandas.core.indexes.api import default_index
from pandas.core.sorting import (
@final
def _maybe_restore_index_levels(self, result: DataFrame) -> None:
    """
        Restore index levels specified as `on` parameters

        Here we check for cases where `self.left_on` and `self.right_on` pairs
        each reference an index level in their respective DataFrames. The
        joined columns corresponding to these pairs are then restored to the
        index of `result`.

        **Note:** This method has side effects. It modifies `result` in-place

        Parameters
        ----------
        result: DataFrame
            merge result

        Returns
        -------
        None
        """
    names_to_restore = []
    for name, left_key, right_key in zip(self.join_names, self.left_on, self.right_on):
        if self.orig_left._is_level_reference(left_key) and self.orig_right._is_level_reference(right_key) and (left_key == right_key) and (name not in result.index.names):
            names_to_restore.append(name)
    if names_to_restore:
        result.set_index(names_to_restore, inplace=True)