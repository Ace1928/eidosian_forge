import textwrap
from copy import copy
import os
from packaging.version import Version
from plotly import optional_imports
from plotly.io._base_renderers import (
from plotly.io._utils import validate_coerce_fig_to_dict
def _available_renderers_str(self):
    """
        Return nicely wrapped string representation of all
        available renderer names
        """
    available = '\n'.join(textwrap.wrap(repr(list(self)), width=79 - 8, initial_indent=' ' * 8, subsequent_indent=' ' * 9))
    return available