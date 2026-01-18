from __future__ import annotations
from abc import (
import sys
from textwrap import dedent
from typing import TYPE_CHECKING
from pandas._config import get_option
from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing
def _get_gross_column_widths(self) -> Sequence[int]:
    """Get widths of columns containing both headers and actual content."""
    body_column_widths = self._get_body_column_widths()
    return [max(*widths) for widths in zip(self.header_column_widths, body_column_widths)]