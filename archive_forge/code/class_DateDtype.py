from __future__ import annotations
import datetime as dt
from typing import (
import numpy as np
from pandas.core.dtypes.dtypes import register_extension_dtype
from pandas.api.extensions import (
from pandas.api.types import pandas_dtype
@register_extension_dtype
class DateDtype(ExtensionDtype):

    @property
    def type(self):
        return dt.date

    @property
    def name(self):
        return 'DateDtype'

    @classmethod
    def construct_from_string(cls, string: str):
        if not isinstance(string, str):
            raise TypeError(f"'construct_from_string' expects a string, got {type(string)}")
        if string == cls.__name__:
            return cls()
        else:
            raise TypeError(f"Cannot construct a '{cls.__name__}' from '{string}'")

    @classmethod
    def construct_array_type(cls):
        return DateArray

    @property
    def na_value(self):
        return dt.date.min

    def __repr__(self) -> str:
        return self.name