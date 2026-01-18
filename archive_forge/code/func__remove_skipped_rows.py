from __future__ import annotations
from collections import (
from collections.abc import (
import csv
from io import StringIO
import re
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas.errors import (
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.inference import is_dict_like
from pandas.io.common import (
from pandas.io.parsers.base_parser import (
def _remove_skipped_rows(self, new_rows: list[list[Scalar]]) -> list[list[Scalar]]:
    if self.skiprows:
        return [row for i, row in enumerate(new_rows) if not self.skipfunc(i + self.pos)]
    return new_rows