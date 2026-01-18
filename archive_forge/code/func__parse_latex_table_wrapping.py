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
def _parse_latex_table_wrapping(table_styles: CSSStyles, caption: str | None) -> bool:
    """
    Indicate whether LaTeX {tabular} should be wrapped with a {table} environment.

    Parses the `table_styles` and detects any selectors which must be included outside
    of {tabular}, i.e. indicating that wrapping must occur, and therefore return True,
    or if a caption exists and requires similar.
    """
    IGNORED_WRAPPERS = ['toprule', 'midrule', 'bottomrule', 'column_format']
    return table_styles is not None and any((d['selector'] not in IGNORED_WRAPPERS for d in table_styles)) or caption is not None