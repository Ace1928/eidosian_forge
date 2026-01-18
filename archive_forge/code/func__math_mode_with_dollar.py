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
def _math_mode_with_dollar(s: str) -> str:
    """
    All characters in LaTeX math mode are preserved.

    The substrings in LaTeX math mode, which start with
    the character ``$`` and end with ``$``, are preserved
    without escaping. Otherwise regular LaTeX escaping applies.

    Parameters
    ----------
    s : str
        Input to be escaped

    Return
    ------
    str :
        Escaped string
    """
    s = s.replace('\\$', 'rt8§=§7wz')
    pattern = re.compile('\\$.*?\\$')
    pos = 0
    ps = pattern.search(s, pos)
    res = []
    while ps:
        res.append(_escape_latex(s[pos:ps.span()[0]]))
        res.append(ps.group())
        pos = ps.span()[1]
        ps = pattern.search(s, pos)
    res.append(_escape_latex(s[pos:len(s)]))
    return ''.join(res).replace('rt8§=§7wz', '\\$')