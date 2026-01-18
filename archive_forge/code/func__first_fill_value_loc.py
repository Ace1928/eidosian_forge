from __future__ import annotations
from collections import abc
import numbers
import operator
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
import pandas._libs.sparse as splib
from pandas._libs.sparse import (
from pandas._libs.tslibs import NaT
from pandas.compat.numpy import function as nv
from pandas.errors import PerformanceWarning
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import (
from pandas.core.dtypes.astype import astype_array
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core import arraylike
import pandas.core.algorithms as algos
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import ExtensionArray
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.indexers import (
from pandas.core.nanops import check_below_min_count
from pandas.io.formats import printing
def _first_fill_value_loc(self):
    """
        Get the location of the first fill value.

        Returns
        -------
        int
        """
    if len(self) == 0 or self.sp_index.npoints == len(self):
        return -1
    indices = self.sp_index.indices
    if not len(indices) or indices[0] > 0:
        return 0
    diff = np.r_[np.diff(indices), 2]
    return indices[(diff > 1).argmax()] + 1