from __future__ import annotations
from abc import (
import sys
from textwrap import dedent
from typing import TYPE_CHECKING
from pandas._config import get_option
from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing
class _SeriesTableBuilderVerbose(_SeriesTableBuilder, _TableBuilderVerboseMixin):
    """
    Series info table builder for verbose output.
    """

    def __init__(self, *, info: SeriesInfo, with_counts: bool) -> None:
        self.info = info
        self.with_counts = with_counts
        self.strrows: Sequence[Sequence[str]] = list(self._gen_rows())
        self.gross_column_widths: Sequence[int] = self._get_gross_column_widths()

    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty series."""
        self.add_object_type_line()
        self.add_index_range_line()
        self.add_series_name_line()
        self.add_header_line()
        self.add_separator_line()
        self.add_body_lines()
        self.add_dtypes_line()
        if self.display_memory_usage:
            self.add_memory_usage_line()

    def add_series_name_line(self) -> None:
        self._lines.append(f'Series name: {self.data.name}')

    @property
    def headers(self) -> Sequence[str]:
        """Headers names of the columns in verbose table."""
        if self.with_counts:
            return ['Non-Null Count', 'Dtype']
        return ['Dtype']

    def _gen_rows_without_counts(self) -> Iterator[Sequence[str]]:
        """Iterator with string representation of body data without counts."""
        yield from self._gen_dtypes()

    def _gen_rows_with_counts(self) -> Iterator[Sequence[str]]:
        """Iterator with string representation of body data with counts."""
        yield from zip(self._gen_non_null_counts(), self._gen_dtypes())