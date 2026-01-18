from __future__ import annotations
import re
from typing import (
import warnings
import numpy as np
from pandas._typing import (
from pandas.errors import PerformanceWarning
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.astype import astype_array
from pandas.core.dtypes.base import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import (
@staticmethod
def _parse_subtype(dtype: str) -> tuple[str, bool]:
    """
        Parse a string to get the subtype

        Parameters
        ----------
        dtype : str
            A string like

            * Sparse[subtype]
            * Sparse[subtype, fill_value]

        Returns
        -------
        subtype : str

        Raises
        ------
        ValueError
            When the subtype cannot be extracted.
        """
    xpr = re.compile('Sparse\\[(?P<subtype>[^,]*)(, )?(?P<fill_value>.*?)?\\]$')
    m = xpr.match(dtype)
    has_fill_value = False
    if m:
        subtype = m.groupdict()['subtype']
        has_fill_value = bool(m.groupdict()['fill_value'])
    elif dtype == 'Sparse':
        subtype = 'float64'
    else:
        raise ValueError(f'Cannot parse {dtype}')
    return (subtype, has_fill_value)