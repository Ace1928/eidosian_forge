from __future__ import annotations
from abc import (
import sys
from textwrap import dedent
from typing import TYPE_CHECKING
from pandas._config import get_option
from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing
def _get_body_column_widths(self) -> Sequence[int]:
    """Get widths of table content columns."""
    strcols: Sequence[Sequence[str]] = list(zip(*self.strrows))
    return [max((len(x) for x in col)) for col in strcols]