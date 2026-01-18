from __future__ import annotations
from abc import (
import sys
from textwrap import dedent
from typing import TYPE_CHECKING
from pandas._config import get_option
from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing
def _gen_rows_with_counts(self) -> Iterator[Sequence[str]]:
    """Iterator with string representation of body data with counts."""
    yield from zip(self._gen_non_null_counts(), self._gen_dtypes())