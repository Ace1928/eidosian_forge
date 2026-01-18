from __future__ import annotations
from collections import defaultdict
from collections.abc import Sequence
from functools import partial
import re
from typing import (
from uuid import uuid4
import numpy as np
from pandas._config import get_option
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.core.dtypes.common import (
from pandas.core.dtypes.generic import ABCSeries
from pandas import (
from pandas.api.types import is_list_like
import pandas.core.common as com
from markupsafe import escape as escape_html  # markupsafe is jinja2 dependency
def _maybe_wrap_formatter(formatter: BaseFormatter | None=None, na_rep: str | None=None, precision: int | None=None, decimal: str='.', thousands: str | None=None, escape: str | None=None, hyperlinks: str | None=None) -> Callable:
    """
    Allows formatters to be expressed as str, callable or None, where None returns
    a default formatting function. wraps with na_rep, and precision where they are
    available.
    """
    if isinstance(formatter, str):
        func_0 = lambda x: formatter.format(x)
    elif callable(formatter):
        func_0 = formatter
    elif formatter is None:
        precision = get_option('styler.format.precision') if precision is None else precision
        func_0 = partial(_default_formatter, precision=precision, thousands=thousands is not None)
    else:
        raise TypeError(f"'formatter' expected str or callable, got {type(formatter)}")
    if escape is not None:
        func_1 = lambda x: func_0(_str_escape(x, escape=escape))
    else:
        func_1 = func_0
    if decimal != '.' or (thousands is not None and thousands != ','):
        func_2 = _wrap_decimal_thousands(func_1, decimal=decimal, thousands=thousands)
    else:
        func_2 = func_1
    if hyperlinks is not None:
        func_3 = lambda x: func_2(_render_href(x, format=hyperlinks))
    else:
        func_3 = func_2
    if na_rep is None:
        return func_3
    else:
        return lambda x: na_rep if isna(x) is True else func_3(x)