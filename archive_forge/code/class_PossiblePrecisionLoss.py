from __future__ import annotations
import ctypes
from pandas._config.config import OptionError
from pandas._libs.tslibs import (
from pandas.util.version import InvalidVersion
class PossiblePrecisionLoss(Warning):
    """
    Warning raised by to_stata on a column with a value outside or equal to int64.

    When the column value is outside or equal to the int64 value the column is
    converted to a float64 dtype.

    Examples
    --------
    >>> df = pd.DataFrame({"s": pd.Series([1, 2**53], dtype=np.int64)})
    >>> df.to_stata('test') # doctest: +SKIP
    ... # PossiblePrecisionLoss: Column converted from int64 to float64...
    """