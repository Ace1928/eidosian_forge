from __future__ import annotations
from collections.abc import (
import itertools
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.internals import (
from pandas._libs.tslibs import Timestamp
from pandas.errors import PerformanceWarning
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import infer_dtype_from_scalar
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
import pandas.core.algorithms as algos
from pandas.core.arrays import (
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.construction import (
from pandas.core.indexers import maybe_convert_indices
from pandas.core.indexes.api import (
from pandas.core.internals.base import (
from pandas.core.internals.blocks import (
from pandas.core.internals.ops import (
def _insert_update_blklocs_and_blknos(self, loc) -> None:
    """
        When inserting a new Block at location 'loc', we update our
        _blklocs and _blknos.
        """
    if loc == self.blklocs.shape[0]:
        self._blklocs = np.append(self._blklocs, 0)
        self._blknos = np.append(self._blknos, len(self.blocks))
    elif loc == 0:
        self._blklocs = np.append(self._blklocs[::-1], 0)[::-1]
        self._blknos = np.append(self._blknos[::-1], len(self.blocks))[::-1]
    else:
        new_blklocs, new_blknos = libinternals.update_blklocs_and_blknos(self.blklocs, self.blknos, loc, len(self.blocks))
        self._blklocs = new_blklocs
        self._blknos = new_blknos