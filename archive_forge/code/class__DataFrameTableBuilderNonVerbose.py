from __future__ import annotations
from abc import (
import sys
from textwrap import dedent
from typing import TYPE_CHECKING
from pandas._config import get_option
from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing
class _DataFrameTableBuilderNonVerbose(_DataFrameTableBuilder):
    """
    Dataframe info table builder for non-verbose output.
    """

    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty dataframe."""
        self.add_object_type_line()
        self.add_index_range_line()
        self.add_columns_summary_line()
        self.add_dtypes_line()
        if self.display_memory_usage:
            self.add_memory_usage_line()

    def add_columns_summary_line(self) -> None:
        self._lines.append(self.ids._summary(name='Columns'))