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
class _Datetime64Formatter(_GenericArrayFormatter):
    values: DatetimeArray

    def __init__(self, values: DatetimeArray, nat_rep: str='NaT', date_format: None=None, **kwargs) -> None:
        super().__init__(values, **kwargs)
        self.nat_rep = nat_rep
        self.date_format = date_format

    def _format_strings(self) -> list[str]:
        """we by definition have DO NOT have a TZ"""
        values = self.values
        if self.formatter is not None:
            return [self.formatter(x) for x in values]
        fmt_values = values._format_native_types(na_rep=self.nat_rep, date_format=self.date_format)
        return fmt_values.tolist()