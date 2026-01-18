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
def _validate_usecols(self, usecols) -> None:
    if lib.is_list_like(usecols) and (not all((isinstance(x, str) for x in usecols))):
        raise ValueError("The pyarrow engine does not allow 'usecols' to be integer column positions. Pass a list of string column names instead.")
    elif callable(usecols):
        raise ValueError("The pyarrow engine does not allow 'usecols' to be a callable.")