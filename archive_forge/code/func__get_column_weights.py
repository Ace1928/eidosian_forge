from __future__ import annotations
from typing import (
import numpy as np
from pandas.core.dtypes.common import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib.core import (
from pandas.plotting._matplotlib.groupby import (
from pandas.plotting._matplotlib.misc import unpack_single_str_list
from pandas.plotting._matplotlib.tools import (
@final
@staticmethod
def _get_column_weights(weights, i: int, y):
    if weights is not None:
        if np.ndim(weights) != 1 and np.shape(weights)[-1] != 1:
            try:
                weights = weights[:, i]
            except IndexError as err:
                raise ValueError('weights must have the same shape as data, or be a single column') from err
        weights = weights[~isna(y)]
    return weights