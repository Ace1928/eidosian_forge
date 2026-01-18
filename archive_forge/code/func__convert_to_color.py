from __future__ import annotations
import mmap
from typing import (
import numpy as np
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import doc
from pandas.core.shared_docs import _shared_docs
from pandas.io.excel._base import (
from pandas.io.excel._util import (
@classmethod
def _convert_to_color(cls, color_spec):
    """
        Convert ``color_spec`` to an openpyxl v2 Color object.

        Parameters
        ----------
        color_spec : str, dict
            A 32-bit ARGB hex string, or a dict with zero or more of the
            following keys.
                'rgb'
                'indexed'
                'auto'
                'theme'
                'tint'
                'index'
                'type'

        Returns
        -------
        color : openpyxl.styles.Color
        """
    from openpyxl.styles import Color
    if isinstance(color_spec, str):
        return Color(color_spec)
    else:
        return Color(**color_spec)