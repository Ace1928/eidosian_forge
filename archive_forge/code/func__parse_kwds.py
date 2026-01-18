from __future__ import annotations
from typing import TYPE_CHECKING
import warnings
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.inference import is_integer
import pandas as pd
from pandas import DataFrame
from pandas.io._util import (
from pandas.io.parsers.base_parser import ParserBase
def _parse_kwds(self) -> None:
    """
        Validates keywords before passing to pyarrow.
        """
    encoding: str | None = self.kwds.get('encoding')
    self.encoding = 'utf-8' if encoding is None else encoding
    na_values = self.kwds['na_values']
    if isinstance(na_values, dict):
        raise ValueError("The pyarrow engine doesn't support passing a dict for na_values")
    self.na_values = list(self.kwds['na_values'])