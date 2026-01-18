from __future__ import annotations
from abc import (
import sys
from textwrap import dedent
from typing import TYPE_CHECKING
from pandas._config import get_option
from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing
def add_body_lines(self) -> None:
    for row in self.strrows:
        body_line = self.SPACING.join([_put_str(col, gross_colwidth) for col, gross_colwidth in zip(row, self.gross_column_widths)])
        self._lines.append(body_line)