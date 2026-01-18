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
def _render_href(x, format):
    """uses regex to detect a common URL pattern and converts to href tag in format."""
    if isinstance(x, str):
        if format == 'html':
            href = '<a href="{0}" target="_blank">{0}</a>'
        elif format == 'latex':
            href = '\\href{{{0}}}{{{0}}}'
        else:
            raise ValueError("``hyperlinks`` format can only be 'html' or 'latex'")
        pat = "((http|ftp)s?:\\/\\/|www.)[\\w/\\-?=%.:@]+\\.[\\w/\\-&?=%.,':;~!@#$*()\\[\\]]+"
        return re.sub(pat, lambda m: href.format(m.group(0)), x)
    return x