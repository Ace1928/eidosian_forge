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
def _sort_labels(uniques: np.ndarray, left: npt.NDArray[np.intp], right: npt.NDArray[np.intp]) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    llength = len(left)
    labels = np.concatenate([left, right])
    _, new_labels = algos.safe_sort(uniques, labels, use_na_sentinel=True)
    new_left, new_right = (new_labels[:llength], new_labels[llength:])
    return (new_left, new_right)