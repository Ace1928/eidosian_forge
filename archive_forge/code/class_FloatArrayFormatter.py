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
class FloatArrayFormatter(_GenericArrayFormatter):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.float_format is not None and self.formatter is None:
            self.fixed_width = False
            if callable(self.float_format):
                self.formatter = self.float_format
                self.float_format = None

    def _value_formatter(self, float_format: FloatFormatType | None=None, threshold: float | None=None) -> Callable:
        """Returns a function to be applied on each value to format it"""
        if float_format is None:
            float_format = self.float_format
        if float_format:

            def base_formatter(v):
                assert float_format is not None
                return float_format(value=v) if notna(v) else self.na_rep
        else:

            def base_formatter(v):
                return str(v) if notna(v) else self.na_rep
        if self.decimal != '.':

            def decimal_formatter(v):
                return base_formatter(v).replace('.', self.decimal, 1)
        else:
            decimal_formatter = base_formatter
        if threshold is None:
            return decimal_formatter

        def formatter(value):
            if notna(value):
                if abs(value) > threshold:
                    return decimal_formatter(value)
                else:
                    return decimal_formatter(0.0)
            else:
                return self.na_rep
        return formatter

    def get_result_as_array(self) -> np.ndarray:
        """
        Returns the float values converted into strings using
        the parameters given at initialisation, as a numpy array
        """

        def format_with_na_rep(values: ArrayLike, formatter: Callable, na_rep: str):
            mask = isna(values)
            formatted = np.array([formatter(val) if not m else na_rep for val, m in zip(values.ravel(), mask.ravel())]).reshape(values.shape)
            return formatted

        def format_complex_with_na_rep(values: ArrayLike, formatter: Callable, na_rep: str):
            real_values = np.real(values).ravel()
            imag_values = np.imag(values).ravel()
            real_mask, imag_mask = (isna(real_values), isna(imag_values))
            formatted_lst = []
            for val, real_val, imag_val, re_isna, im_isna in zip(values.ravel(), real_values, imag_values, real_mask, imag_mask):
                if not re_isna and (not im_isna):
                    formatted_lst.append(formatter(val))
                elif not re_isna:
                    formatted_lst.append(f'{formatter(real_val)}+{na_rep}j')
                elif not im_isna:
                    imag_formatted = formatter(imag_val).strip()
                    if imag_formatted.startswith('-'):
                        formatted_lst.append(f'{na_rep}{imag_formatted}j')
                    else:
                        formatted_lst.append(f'{na_rep}+{imag_formatted}j')
                else:
                    formatted_lst.append(f'{na_rep}+{na_rep}j')
            return np.array(formatted_lst).reshape(values.shape)
        if self.formatter is not None:
            return format_with_na_rep(self.values, self.formatter, self.na_rep)
        if self.fixed_width:
            threshold = get_option('display.chop_threshold')
        else:
            threshold = None

        def format_values_with(float_format):
            formatter = self._value_formatter(float_format, threshold)
            na_rep = ' ' + self.na_rep if self.justify == 'left' else self.na_rep
            values = self.values
            is_complex = is_complex_dtype(values)
            if is_complex:
                values = format_complex_with_na_rep(values, formatter, na_rep)
            else:
                values = format_with_na_rep(values, formatter, na_rep)
            if self.fixed_width:
                if is_complex:
                    result = _trim_zeros_complex(values, self.decimal)
                else:
                    result = _trim_zeros_float(values, self.decimal)
                return np.asarray(result, dtype='object')
            return values
        float_format: FloatFormatType | None
        if self.float_format is None:
            if self.fixed_width:
                if self.leading_space is True:
                    fmt_str = '{value: .{digits:d}f}'
                else:
                    fmt_str = '{value:.{digits:d}f}'
                float_format = partial(fmt_str.format, digits=self.digits)
            else:
                float_format = self.float_format
        else:
            float_format = lambda value: self.float_format % value
        formatted_values = format_values_with(float_format)
        if not self.fixed_width:
            return formatted_values
        if len(formatted_values) > 0:
            maxlen = max((len(x) for x in formatted_values))
            too_long = maxlen > self.digits + 6
        else:
            too_long = False
        abs_vals = np.abs(self.values)
        has_large_values = (abs_vals > 1000000.0).any()
        has_small_values = ((abs_vals < 10 ** (-self.digits)) & (abs_vals > 0)).any()
        if has_small_values or (too_long and has_large_values):
            if self.leading_space is True:
                fmt_str = '{value: .{digits:d}e}'
            else:
                fmt_str = '{value:.{digits:d}e}'
            float_format = partial(fmt_str.format, digits=self.digits)
            formatted_values = format_values_with(float_format)
        return formatted_values

    def _format_strings(self) -> list[str]:
        return list(self.get_result_as_array())