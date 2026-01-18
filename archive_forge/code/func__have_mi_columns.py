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
@cache_readonly
def _have_mi_columns(self) -> bool:
    if self.header is None:
        return False
    header = self.header
    if isinstance(header, (list, tuple, np.ndarray)):
        return len(header) > 1
    else:
        return False