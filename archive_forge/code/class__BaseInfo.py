from __future__ import annotations
from abc import (
import sys
from textwrap import dedent
from typing import TYPE_CHECKING
from pandas._config import get_option
from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing
class _BaseInfo(ABC):
    """
    Base class for DataFrameInfo and SeriesInfo.

    Parameters
    ----------
    data : DataFrame or Series
        Either dataframe or series.
    memory_usage : bool or str, optional
        If "deep", introspect the data deeply by interrogating object dtypes
        for system-level memory consumption, and include it in the returned
        values.
    """
    data: DataFrame | Series
    memory_usage: bool | str

    @property
    @abstractmethod
    def dtypes(self) -> Iterable[Dtype]:
        """
        Dtypes.

        Returns
        -------
        dtypes : sequence
            Dtype of each of the DataFrame's columns (or one series column).
        """

    @property
    @abstractmethod
    def dtype_counts(self) -> Mapping[str, int]:
        """Mapping dtype - number of counts."""

    @property
    @abstractmethod
    def non_null_counts(self) -> Sequence[int]:
        """Sequence of non-null counts for all columns or column (if series)."""

    @property
    @abstractmethod
    def memory_usage_bytes(self) -> int:
        """
        Memory usage in bytes.

        Returns
        -------
        memory_usage_bytes : int
            Object's total memory usage in bytes.
        """

    @property
    def memory_usage_string(self) -> str:
        """Memory usage in a form of human readable string."""
        return f'{_sizeof_fmt(self.memory_usage_bytes, self.size_qualifier)}\n'

    @property
    def size_qualifier(self) -> str:
        size_qualifier = ''
        if self.memory_usage:
            if self.memory_usage != 'deep':
                if 'object' in self.dtype_counts or self.data.index._is_memory_usage_qualified():
                    size_qualifier = '+'
        return size_qualifier

    @abstractmethod
    def render(self, *, buf: WriteBuffer[str] | None, max_cols: int | None, verbose: bool | None, show_counts: bool | None) -> None:
        pass