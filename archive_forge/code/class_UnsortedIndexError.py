from __future__ import annotations
import ctypes
from pandas._config.config import OptionError
from pandas._libs.tslibs import (
from pandas.util.version import InvalidVersion
class UnsortedIndexError(KeyError):
    """
    Error raised when slicing a MultiIndex which has not been lexsorted.

    Subclass of `KeyError`.

    Examples
    --------
    >>> df = pd.DataFrame({"cat": [0, 0, 1, 1],
    ...                    "color": ["white", "white", "brown", "black"],
    ...                    "lives": [4, 4, 3, 7]},
    ...                   )
    >>> df = df.set_index(["cat", "color"])
    >>> df
                lives
    cat  color
    0    white    4
         white    4
    1    brown    3
         black    7
    >>> df.loc[(0, "black"):(1, "white")]
    Traceback (most recent call last):
    UnsortedIndexError: 'Key length (2) was greater
    than MultiIndex lexsort depth (1)'
    """