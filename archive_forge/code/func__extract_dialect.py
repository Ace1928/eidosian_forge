from __future__ import annotations
from collections import (
import csv
import sys
from textwrap import fill
from typing import (
import warnings
import numpy as np
from pandas._config import using_copy_on_write
from pandas._libs import lib
from pandas._libs.parsers import STR_NA_VALUES
from pandas.errors import (
from pandas.util._decorators import Appender
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend
from pandas.core.dtypes.common import (
from pandas import Series
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import RangeIndex
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import (
from pandas.io.parsers.arrow_parser_wrapper import ArrowParserWrapper
from pandas.io.parsers.base_parser import (
from pandas.io.parsers.c_parser_wrapper import CParserWrapper
from pandas.io.parsers.python_parser import (
def _extract_dialect(kwds: dict[str, Any]) -> csv.Dialect | None:
    """
    Extract concrete csv dialect instance.

    Returns
    -------
    csv.Dialect or None
    """
    if kwds.get('dialect') is None:
        return None
    dialect = kwds['dialect']
    if dialect in csv.list_dialects():
        dialect = csv.get_dialect(dialect)
    _validate_dialect(dialect)
    return dialect