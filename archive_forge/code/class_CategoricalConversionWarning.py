from __future__ import annotations
import ctypes
from pandas._config.config import OptionError
from pandas._libs.tslibs import (
from pandas.util.version import InvalidVersion
class CategoricalConversionWarning(Warning):
    """
    Warning is raised when reading a partial labeled Stata file using a iterator.

    Examples
    --------
    >>> from pandas.io.stata import StataReader
    >>> with StataReader('dta_file', chunksize=2) as reader: # doctest: +SKIP
    ...   for i, block in enumerate(reader):
    ...      print(i, block)
    ... # CategoricalConversionWarning: One or more series with value labels...
    """