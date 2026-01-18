from __future__ import annotations
from abc import (
import sys
from textwrap import dedent
from typing import TYPE_CHECKING
from pandas._config import get_option
from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing
def _fill_empty_info(self) -> None:
    """Add lines to the info table, pertaining to empty dataframe."""
    self.add_object_type_line()
    self.add_index_range_line()
    self._lines.append(f'Empty {type(self.data).__name__}\n')