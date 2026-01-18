from __future__ import annotations
from abc import (
import sys
from textwrap import dedent
from typing import TYPE_CHECKING
from pandas._config import get_option
from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing
class _DataFrameInfoPrinter(_InfoPrinterAbstract):
    """
    Class for printing dataframe info.

    Parameters
    ----------
    info : DataFrameInfo
        Instance of DataFrameInfo.
    max_cols : int, optional
        When to switch from the verbose to the truncated output.
    verbose : bool, optional
        Whether to print the full summary.
    show_counts : bool, optional
        Whether to show the non-null counts.
    """

    def __init__(self, info: DataFrameInfo, max_cols: int | None=None, verbose: bool | None=None, show_counts: bool | None=None) -> None:
        self.info = info
        self.data = info.data
        self.verbose = verbose
        self.max_cols = self._initialize_max_cols(max_cols)
        self.show_counts = self._initialize_show_counts(show_counts)

    @property
    def max_rows(self) -> int:
        """Maximum info rows to be displayed."""
        return get_option('display.max_info_rows', len(self.data) + 1)

    @property
    def exceeds_info_cols(self) -> bool:
        """Check if number of columns to be summarized does not exceed maximum."""
        return bool(self.col_count > self.max_cols)

    @property
    def exceeds_info_rows(self) -> bool:
        """Check if number of rows to be summarized does not exceed maximum."""
        return bool(len(self.data) > self.max_rows)

    @property
    def col_count(self) -> int:
        """Number of columns to be summarized."""
        return self.info.col_count

    def _initialize_max_cols(self, max_cols: int | None) -> int:
        if max_cols is None:
            return get_option('display.max_info_columns', self.col_count + 1)
        return max_cols

    def _initialize_show_counts(self, show_counts: bool | None) -> bool:
        if show_counts is None:
            return bool(not self.exceeds_info_cols and (not self.exceeds_info_rows))
        else:
            return show_counts

    def _create_table_builder(self) -> _DataFrameTableBuilder:
        """
        Create instance of table builder based on verbosity and display settings.
        """
        if self.verbose:
            return _DataFrameTableBuilderVerbose(info=self.info, with_counts=self.show_counts)
        elif self.verbose is False:
            return _DataFrameTableBuilderNonVerbose(info=self.info)
        elif self.exceeds_info_cols:
            return _DataFrameTableBuilderNonVerbose(info=self.info)
        else:
            return _DataFrameTableBuilderVerbose(info=self.info, with_counts=self.show_counts)