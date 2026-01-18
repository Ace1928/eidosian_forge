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
def _get_options_with_defaults(self, engine: CSVEngine) -> dict[str, Any]:
    kwds = self.orig_options
    options = {}
    default: object | None
    for argname, default in parser_defaults.items():
        value = kwds.get(argname, default)
        if engine == 'pyarrow' and argname in _pyarrow_unsupported and (value != default) and (value != getattr(value, 'value', default)):
            raise ValueError(f"The {repr(argname)} option is not supported with the 'pyarrow' engine")
        options[argname] = value
    for argname, default in _c_parser_defaults.items():
        if argname in kwds:
            value = kwds[argname]
            if engine != 'c' and value != default:
                if 'python' in engine and argname not in _python_unsupported:
                    pass
                elif 'pyarrow' in engine and argname not in _pyarrow_unsupported:
                    pass
                else:
                    raise ValueError(f'The {repr(argname)} option is not supported with the {repr(engine)} engine')
        else:
            value = default
        options[argname] = value
    if engine == 'python-fwf':
        for argname, default in _fwf_defaults.items():
            options[argname] = kwds.get(argname, default)
    return options