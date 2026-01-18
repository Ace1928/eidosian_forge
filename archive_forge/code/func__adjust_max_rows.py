from __future__ import annotations
from collections.abc import (
from contextlib import contextmanager
from csv import QUOTE_NONE
from decimal import Decimal
from functools import partial
from io import StringIO
import math
import re
from shutil import get_terminal_size
from typing import (
import numpy as np
from pandas._config.config import (
from pandas._libs import lib
from pandas._libs.missing import NA
from pandas._libs.tslibs import (
from pandas._libs.tslibs.nattype import NaTType
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import (
from pandas.core.arrays import (
from pandas.core.arrays.string_ import StringDtype
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.indexes.api import (
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.core.reshape.concat import concat
from pandas.io.common import (
from pandas.io.formats import printing
def _adjust_max_rows(self, max_rows: int | None) -> int | None:
    """Adjust max_rows using display logic.

        See description here:
        https://pandas.pydata.org/docs/dev/user_guide/options.html#frequently-used-options

        GH #37359
        """
    if max_rows:
        if len(self.frame) > max_rows and self.min_rows:
            max_rows = min(self.min_rows, max_rows)
    return max_rows