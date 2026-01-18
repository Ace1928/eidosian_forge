from __future__ import annotations
from abc import (
import sys
from textwrap import dedent
from typing import TYPE_CHECKING
from pandas._config import get_option
from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing
def _create_table_builder(self) -> _SeriesTableBuilder:
    """
        Create instance of table builder based on verbosity.
        """
    if self.verbose or self.verbose is None:
        return _SeriesTableBuilderVerbose(info=self.info, with_counts=self.show_counts)
    else:
        return _SeriesTableBuilderNonVerbose(info=self.info)