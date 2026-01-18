from __future__ import annotations
import ctypes
from pandas._config.config import OptionError
from pandas._libs.tslibs import (
from pandas.util.version import InvalidVersion
class IndexingError(Exception):
    """
    Exception is raised when trying to index and there is a mismatch in dimensions.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 1, 1]})
    >>> df.loc[..., ..., 'A'] # doctest: +SKIP
    ... # IndexingError: indexer may only contain one '...' entry
    >>> df = pd.DataFrame({'A': [1, 1, 1]})
    >>> df.loc[1, ..., ...] # doctest: +SKIP
    ... # IndexingError: Too many indexers
    >>> df[pd.Series([True], dtype=bool)] # doctest: +SKIP
    ... # IndexingError: Unalignable boolean Series provided as indexer...
    >>> s = pd.Series(range(2),
    ...               index = pd.MultiIndex.from_product([["a", "b"], ["c"]]))
    >>> s.loc["a", "c", "d"] # doctest: +SKIP
    ... # IndexingError: Too many indexers
    """