from __future__ import annotations
import ctypes
from pandas._config.config import OptionError
from pandas._libs.tslibs import (
from pandas.util.version import InvalidVersion
class EmptyDataError(ValueError):
    """
    Exception raised in ``pd.read_csv`` when empty data or header is encountered.

    Examples
    --------
    >>> from io import StringIO
    >>> empty = StringIO()
    >>> pd.read_csv(empty)
    Traceback (most recent call last):
    EmptyDataError: No columns to parse from file
    """