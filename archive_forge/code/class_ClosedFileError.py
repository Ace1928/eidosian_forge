from __future__ import annotations
import ctypes
from pandas._config.config import OptionError
from pandas._libs.tslibs import (
from pandas.util.version import InvalidVersion
class ClosedFileError(Exception):
    """
    Exception is raised when trying to perform an operation on a closed HDFStore file.

    Examples
    --------
    >>> store = pd.HDFStore('my-store', 'a') # doctest: +SKIP
    >>> store.close() # doctest: +SKIP
    >>> store.keys() # doctest: +SKIP
    ... # ClosedFileError: my-store file is not open!
    """