from __future__ import annotations
from abc import (
import sys
from textwrap import dedent
from typing import TYPE_CHECKING
from pandas._config import get_option
from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing
class _SeriesInfoPrinter(_InfoPrinterAbstract):
    """Class for printing series info.

    Parameters
    ----------
    info : SeriesInfo
        Instance of SeriesInfo.
    verbose : bool, optional
        Whether to print the full summary.
    show_counts : bool, optional
        Whether to show the non-null counts.
    """

    def __init__(self, info: SeriesInfo, verbose: bool | None=None, show_counts: bool | None=None) -> None:
        self.info = info
        self.data = info.data
        self.verbose = verbose
        self.show_counts = self._initialize_show_counts(show_counts)

    def _create_table_builder(self) -> _SeriesTableBuilder:
        """
        Create instance of table builder based on verbosity.
        """
        if self.verbose or self.verbose is None:
            return _SeriesTableBuilderVerbose(info=self.info, with_counts=self.show_counts)
        else:
            return _SeriesTableBuilderNonVerbose(info=self.info)

    def _initialize_show_counts(self, show_counts: bool | None) -> bool:
        if show_counts is None:
            return True
        else:
            return show_counts