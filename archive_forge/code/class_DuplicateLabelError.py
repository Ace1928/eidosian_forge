from __future__ import annotations
import ctypes
from pandas._config.config import OptionError
from pandas._libs.tslibs import (
from pandas.util.version import InvalidVersion
class DuplicateLabelError(ValueError):
    """
    Error raised when an operation would introduce duplicate labels.

    Examples
    --------
    >>> s = pd.Series([0, 1, 2], index=['a', 'b', 'c']).set_flags(
    ...     allows_duplicate_labels=False
    ... )
    >>> s.reindex(['a', 'a', 'b'])
    Traceback (most recent call last):
       ...
    DuplicateLabelError: Index has duplicates.
          positions
    label
    a        [0, 1]
    """