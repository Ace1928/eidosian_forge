from __future__ import annotations
import itertools
from typing import (
import numpy as np
from pandas._libs import (
from pandas.core.dtypes.astype import (
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
import pandas.core.algorithms as algos
from pandas.core.array_algos.quantile import quantile_compat
from pandas.core.array_algos.take import take_1d
from pandas.core.arrays import (
from pandas.core.construction import (
from pandas.core.indexers import (
from pandas.core.indexes.api import (
from pandas.core.indexes.base import get_values_for_csv
from pandas.core.internals.base import (
from pandas.core.internals.blocks import (
from pandas.core.internals.managers import make_na_array

        Helper function to create the actual all-NA array from the NullArrayProxy
        object.

        Parameters
        ----------
        arr : NullArrayProxy
        dtype : the dtype for the resulting array

        Returns
        -------
        np.ndarray or ExtensionArray
        