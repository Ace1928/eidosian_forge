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
def _convert_to_number_format(cls, number_format_dict):
    """
        Convert ``number_format_dict`` to an openpyxl v2.1.0 number format
        initializer.

        Parameters
        ----------
        number_format_dict : dict
            A dict with zero or more of the following keys.
                'format_code' : str

        Returns
        -------
        number_format : str
        """
    return number_format_dict['format_code']