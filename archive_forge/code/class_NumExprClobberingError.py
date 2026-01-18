from __future__ import annotations
import ctypes
from pandas._config.config import OptionError
from pandas._libs.tslibs import (
from pandas.util.version import InvalidVersion
class NumExprClobberingError(NameError):
    """
    Exception raised when trying to use a built-in numexpr name as a variable name.

    ``eval`` or ``query`` will throw the error if the engine is set
    to 'numexpr'. 'numexpr' is the default engine value for these methods if the
    numexpr package is installed.

    Examples
    --------
    >>> df = pd.DataFrame({'abs': [1, 1, 1]})
    >>> df.query("abs > 2") # doctest: +SKIP
    ... # NumExprClobberingError: Variables in expression "(abs) > (2)" overlap...
    >>> sin, a = 1, 2
    >>> pd.eval("sin + a", engine='numexpr') # doctest: +SKIP
    ... # NumExprClobberingError: Variables in expression "(sin) + (a)" overlap...
    """