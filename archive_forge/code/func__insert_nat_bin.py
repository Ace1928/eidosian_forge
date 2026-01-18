from __future__ import annotations
import copy
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
from pandas._typing import NDFrameT
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError
from pandas.util._decorators import (
from pandas.util._exceptions import (
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas.core.dtypes.generic import (
import pandas.core.algorithms as algos
from pandas.core.apply import (
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.base import (
import pandas.core.common as com
from pandas.core.generic import (
from pandas.core.groupby.generic import SeriesGroupBy
from pandas.core.groupby.groupby import (
from pandas.core.groupby.grouper import Grouper
from pandas.core.groupby.ops import BinGrouper
from pandas.core.indexes.api import MultiIndex
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimes import (
from pandas.core.indexes.period import (
from pandas.core.indexes.timedeltas import (
from pandas.tseries.frequencies import (
from pandas.tseries.offsets import (
def _insert_nat_bin(binner: PeriodIndex, bins: np.ndarray, labels: PeriodIndex, nat_count: int) -> tuple[PeriodIndex, np.ndarray, PeriodIndex]:
    assert nat_count > 0
    bins += nat_count
    bins = np.insert(bins, 0, nat_count)
    binner = binner.insert(0, NaT)
    labels = labels.insert(0, NaT)
    return (binner, bins, labels)