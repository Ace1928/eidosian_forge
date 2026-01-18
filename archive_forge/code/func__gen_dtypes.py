from __future__ import annotations
from abc import (
import sys
from textwrap import dedent
from typing import TYPE_CHECKING
from pandas._config import get_option
from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing
def _gen_dtypes(self) -> Iterator[str]:
    """Iterator with string representation of column dtypes."""
    for dtype in self.dtypes:
        yield pprint_thing(dtype)