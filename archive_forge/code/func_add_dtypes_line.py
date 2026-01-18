from __future__ import annotations
from abc import (
import sys
from textwrap import dedent
from typing import TYPE_CHECKING
from pandas._config import get_option
from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing
def add_dtypes_line(self) -> None:
    """Add summary line with dtypes present in dataframe."""
    collected_dtypes = [f'{key}({val:d})' for key, val in sorted(self.dtype_counts.items())]
    self._lines.append(f'dtypes: {', '.join(collected_dtypes)}')