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
def _parse_latex_header_span(cell: dict[str, Any], multirow_align: str, multicol_align: str, wrap: bool=False, convert_css: bool=False) -> str:
    """
    Refactor the cell `display_value` if a 'colspan' or 'rowspan' attribute is present.

    'rowspan' and 'colspan' do not occur simultaneouly. If they are detected then
    the `display_value` is altered to a LaTeX `multirow` or `multicol` command
    respectively, with the appropriate cell-span.

    ``wrap`` is used to enclose the `display_value` in braces which is needed for
    column headers using an siunitx package.

    Requires the package {multirow}, whereas multicol support is usually built in
    to the {tabular} environment.

    Examples
    --------
    >>> cell = {'cellstyle': '', 'display_value':'text', 'attributes': 'colspan="3"'}
    >>> _parse_latex_header_span(cell, 't', 'c')
    '\\\\multicolumn{3}{c}{text}'
    """
    display_val = _parse_latex_cell_styles(cell['cellstyle'], cell['display_value'], convert_css)
    if 'attributes' in cell:
        attrs = cell['attributes']
        if 'colspan="' in attrs:
            colspan = attrs[attrs.find('colspan="') + 9:]
            colspan = int(colspan[:colspan.find('"')])
            if 'naive-l' == multicol_align:
                out = f'{{{display_val}}}' if wrap else f'{display_val}'
                blanks = ' & {}' if wrap else ' &'
                return out + blanks * (colspan - 1)
            elif 'naive-r' == multicol_align:
                out = f'{{{display_val}}}' if wrap else f'{display_val}'
                blanks = '{} & ' if wrap else '& '
                return blanks * (colspan - 1) + out
            return f'\\multicolumn{{{colspan}}}{{{multicol_align}}}{{{display_val}}}'
        elif 'rowspan="' in attrs:
            if multirow_align == 'naive':
                return display_val
            rowspan = attrs[attrs.find('rowspan="') + 9:]
            rowspan = int(rowspan[:rowspan.find('"')])
            return f'\\multirow[{multirow_align}]{{{rowspan}}}{{*}}{{{display_val}}}'
    if wrap:
        return f'{{{display_val}}}'
    else:
        return display_val