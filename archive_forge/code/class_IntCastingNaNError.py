from __future__ import annotations
import ctypes
from pandas._config.config import OptionError
from pandas._libs.tslibs import (
from pandas.util.version import InvalidVersion
class IntCastingNaNError(ValueError):
    """
    Exception raised when converting (``astype``) an array with NaN to an integer type.

    Examples
    --------
    >>> pd.DataFrame(np.array([[1, np.nan], [2, 3]]), dtype="i8")
    Traceback (most recent call last):
    IntCastingNaNError: Cannot convert non-finite values (NA or inf) to integer
    """