from __future__ import annotations
from abc import (
import sys
from textwrap import dedent
from typing import TYPE_CHECKING
from pandas._config import get_option
from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing
def _gen_rows(self) -> Iterator[Sequence[str]]:
    """
        Generator function yielding rows content.

        Each element represents a row comprising a sequence of strings.
        """
    if self.with_counts:
        return self._gen_rows_with_counts()
    else:
        return self._gen_rows_without_counts()