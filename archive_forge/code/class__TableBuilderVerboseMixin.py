from __future__ import annotations
from abc import (
import sys
from textwrap import dedent
from typing import TYPE_CHECKING
from pandas._config import get_option
from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing
class _TableBuilderVerboseMixin(_TableBuilderAbstract):
    """
    Mixin for verbose info output.
    """
    SPACING: str = ' ' * 2
    strrows: Sequence[Sequence[str]]
    gross_column_widths: Sequence[int]
    with_counts: bool

    @property
    @abstractmethod
    def headers(self) -> Sequence[str]:
        """Headers names of the columns in verbose table."""

    @property
    def header_column_widths(self) -> Sequence[int]:
        """Widths of header columns (only titles)."""
        return [len(col) for col in self.headers]

    def _get_gross_column_widths(self) -> Sequence[int]:
        """Get widths of columns containing both headers and actual content."""
        body_column_widths = self._get_body_column_widths()
        return [max(*widths) for widths in zip(self.header_column_widths, body_column_widths)]

    def _get_body_column_widths(self) -> Sequence[int]:
        """Get widths of table content columns."""
        strcols: Sequence[Sequence[str]] = list(zip(*self.strrows))
        return [max((len(x) for x in col)) for col in strcols]

    def _gen_rows(self) -> Iterator[Sequence[str]]:
        """
        Generator function yielding rows content.

        Each element represents a row comprising a sequence of strings.
        """
        if self.with_counts:
            return self._gen_rows_with_counts()
        else:
            return self._gen_rows_without_counts()

    @abstractmethod
    def _gen_rows_with_counts(self) -> Iterator[Sequence[str]]:
        """Iterator with string representation of body data with counts."""

    @abstractmethod
    def _gen_rows_without_counts(self) -> Iterator[Sequence[str]]:
        """Iterator with string representation of body data without counts."""

    def add_header_line(self) -> None:
        header_line = self.SPACING.join([_put_str(header, col_width) for header, col_width in zip(self.headers, self.gross_column_widths)])
        self._lines.append(header_line)

    def add_separator_line(self) -> None:
        separator_line = self.SPACING.join([_put_str('-' * header_colwidth, gross_colwidth) for header_colwidth, gross_colwidth in zip(self.header_column_widths, self.gross_column_widths)])
        self._lines.append(separator_line)

    def add_body_lines(self) -> None:
        for row in self.strrows:
            body_line = self.SPACING.join([_put_str(col, gross_colwidth) for col, gross_colwidth in zip(row, self.gross_column_widths)])
            self._lines.append(body_line)

    def _gen_non_null_counts(self) -> Iterator[str]:
        """Iterator with string representation of non-null counts."""
        for count in self.non_null_counts:
            yield f'{count} non-null'

    def _gen_dtypes(self) -> Iterator[str]:
        """Iterator with string representation of column dtypes."""
        for dtype in self.dtypes:
            yield pprint_thing(dtype)