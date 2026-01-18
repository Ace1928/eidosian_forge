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
class _GenericArrayFormatter:

    def __init__(self, values: ArrayLike, digits: int=7, formatter: Callable | None=None, na_rep: str='NaN', space: str | int=12, float_format: FloatFormatType | None=None, justify: str='right', decimal: str='.', quoting: int | None=None, fixed_width: bool=True, leading_space: bool | None=True, fallback_formatter: Callable | None=None) -> None:
        self.values = values
        self.digits = digits
        self.na_rep = na_rep
        self.space = space
        self.formatter = formatter
        self.float_format = float_format
        self.justify = justify
        self.decimal = decimal
        self.quoting = quoting
        self.fixed_width = fixed_width
        self.leading_space = leading_space
        self.fallback_formatter = fallback_formatter

    def get_result(self) -> list[str]:
        fmt_values = self._format_strings()
        return _make_fixed_width(fmt_values, self.justify)

    def _format_strings(self) -> list[str]:
        if self.float_format is None:
            float_format = get_option('display.float_format')
            if float_format is None:
                precision = get_option('display.precision')
                float_format = lambda x: _trim_zeros_single_float(f'{x: .{precision:d}f}')
        else:
            float_format = self.float_format
        if self.formatter is not None:
            formatter = self.formatter
        elif self.fallback_formatter is not None:
            formatter = self.fallback_formatter
        else:
            quote_strings = self.quoting is not None and self.quoting != QUOTE_NONE
            formatter = partial(printing.pprint_thing, escape_chars=('\t', '\r', '\n'), quote_strings=quote_strings)

        def _format(x):
            if self.na_rep is not None and is_scalar(x) and isna(x):
                if x is None:
                    return 'None'
                elif x is NA:
                    return str(NA)
                elif lib.is_float(x) and np.isinf(x):
                    return str(x)
                elif x is NaT or isinstance(x, (np.datetime64, np.timedelta64)):
                    return 'NaT'
                return self.na_rep
            elif isinstance(x, PandasObject):
                return str(x)
            elif isinstance(x, StringDtype):
                return repr(x)
            else:
                return str(formatter(x))
        vals = self.values
        if not isinstance(vals, np.ndarray):
            raise TypeError('ExtensionArray formatting should use _ExtensionArrayFormatter')
        inferred = lib.map_infer(vals, is_float)
        is_float_type = inferred & np.all(notna(vals), axis=tuple(range(1, len(vals.shape))))
        leading_space = self.leading_space
        if leading_space is None:
            leading_space = is_float_type.any()
        fmt_values = []
        for i, v in enumerate(vals):
            if (not is_float_type[i] or self.formatter is not None) and leading_space:
                fmt_values.append(f' {_format(v)}')
            elif is_float_type[i]:
                fmt_values.append(float_format(v))
            else:
                if leading_space is False:
                    tpl = '{v}'
                else:
                    tpl = ' {v}'
                fmt_values.append(tpl.format(v=_format(v)))
        return fmt_values