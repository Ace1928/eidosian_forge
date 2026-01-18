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
def _merge_with_dialect_properties(dialect: csv.Dialect, defaults: dict[str, Any]) -> dict[str, Any]:
    """
    Merge default kwargs in TextFileReader with dialect parameters.

    Parameters
    ----------
    dialect : csv.Dialect
        Concrete csv dialect. See csv.Dialect documentation for more details.
    defaults : dict
        Keyword arguments passed to TextFileReader.

    Returns
    -------
    kwds : dict
        Updated keyword arguments, merged with dialect parameters.
    """
    kwds = defaults.copy()
    for param in MANDATORY_DIALECT_ATTRS:
        dialect_val = getattr(dialect, param)
        parser_default = parser_defaults[param]
        provided = kwds.get(param, parser_default)
        conflict_msgs = []
        if provided not in (parser_default, dialect_val):
            msg = f"Conflicting values for '{param}': '{provided}' was provided, but the dialect specifies '{dialect_val}'. Using the dialect-specified value."
            if not (param == 'delimiter' and kwds.pop('sep_override', False)):
                conflict_msgs.append(msg)
        if conflict_msgs:
            warnings.warn('\n\n'.join(conflict_msgs), ParserWarning, stacklevel=find_stack_level())
        kwds[param] = dialect_val
    return kwds