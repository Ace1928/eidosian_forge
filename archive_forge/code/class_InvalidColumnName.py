from __future__ import annotations
import ctypes
from pandas._config.config import OptionError
from pandas._libs.tslibs import (
from pandas.util.version import InvalidVersion
class InvalidColumnName(Warning):
    """
    Warning raised by to_stata the column contains a non-valid stata name.

    Because the column name is an invalid Stata variable, the name needs to be
    converted.

    Examples
    --------
    >>> df = pd.DataFrame({"0categories": pd.Series([2, 2])})
    >>> df.to_stata('test') # doctest: +SKIP
    ... # InvalidColumnName: Not all pandas column names were valid Stata variable...
    """