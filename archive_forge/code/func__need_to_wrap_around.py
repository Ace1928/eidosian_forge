from __future__ import annotations
from shutil import get_terminal_size
from typing import TYPE_CHECKING
import numpy as np
from pandas.io.formats.printing import pprint_thing
@property
def _need_to_wrap_around(self) -> bool:
    return bool(self.fmt.max_cols is None or self.fmt.max_cols > 0)