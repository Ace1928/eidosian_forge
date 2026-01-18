from __future__ import annotations
import ctypes
from pandas._config.config import OptionError
from pandas._libs.tslibs import (
from pandas.util.version import InvalidVersion
class PossibleDataLossError(Exception):
    """
    Exception raised when trying to open a HDFStore file when already opened.

    Examples
    --------
    >>> store = pd.HDFStore('my-store', 'a') # doctest: +SKIP
    >>> store.open("w") # doctest: +SKIP
    ... # PossibleDataLossError: Re-opening the file [my-store] with mode [a]...
    """