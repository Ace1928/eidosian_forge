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
def _element(html_element: str, html_class: str | None, value: Any, is_visible: bool, **kwargs) -> dict:
    """
    Template to return container with information for a <td></td> or <th></th> element.
    """
    if 'display_value' not in kwargs:
        kwargs['display_value'] = value
    return {'type': html_element, 'value': value, 'class': html_class, 'is_visible': is_visible, **kwargs}