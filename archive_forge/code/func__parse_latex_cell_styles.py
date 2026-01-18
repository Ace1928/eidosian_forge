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
def _parse_latex_cell_styles(latex_styles: CSSList, display_value: str, convert_css: bool=False) -> str:
    """
    Mutate the ``display_value`` string including LaTeX commands from ``latex_styles``.

    This method builds a recursive latex chain of commands based on the
    CSSList input, nested around ``display_value``.

    If a CSS style is given as ('<command>', '<options>') this is translated to
    '\\<command><options>{display_value}', and this value is treated as the
    display value for the next iteration.

    The most recent style forms the inner component, for example for styles:
    `[('c1', 'o1'), ('c2', 'o2')]` this returns: `\\c1o1{\\c2o2{display_value}}`

    Sometimes latex commands have to be wrapped with curly braces in different ways:
    We create some parsing flags to identify the different behaviours:

     - `--rwrap`        : `\\<command><options>{<display_value>}`
     - `--wrap`         : `{\\<command><options> <display_value>}`
     - `--nowrap`       : `\\<command><options> <display_value>`
     - `--lwrap`        : `{\\<command><options>} <display_value>`
     - `--dwrap`        : `{\\<command><options>}{<display_value>}`

    For example for styles:
    `[('c1', 'o1--wrap'), ('c2', 'o2')]` this returns: `{\\c1o1 \\c2o2{display_value}}
    """
    if convert_css:
        latex_styles = _parse_latex_css_conversion(latex_styles)
    for command, options in latex_styles[::-1]:
        formatter = {'--wrap': f'{{\\{command}--to_parse {display_value}}}', '--nowrap': f'\\{command}--to_parse {display_value}', '--lwrap': f'{{\\{command}--to_parse}} {display_value}', '--rwrap': f'\\{command}--to_parse{{{display_value}}}', '--dwrap': f'{{\\{command}--to_parse}}{{{display_value}}}'}
        display_value = f'\\{command}{options} {display_value}'
        for arg in ['--nowrap', '--wrap', '--lwrap', '--rwrap', '--dwrap']:
            if arg in str(options):
                display_value = formatter[arg].replace('--to_parse', _parse_latex_options_strip(value=options, arg=arg))
                break
    return display_value