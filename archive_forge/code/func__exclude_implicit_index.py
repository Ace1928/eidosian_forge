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
def _exclude_implicit_index(self, alldata: list[np.ndarray]) -> tuple[Mapping[Hashable, np.ndarray], Sequence[Hashable]]:
    names = dedup_names(self.orig_names, is_potential_multi_index(self.orig_names, self.index_col))
    offset = 0
    if self._implicit_index:
        offset = len(self.index_col)
    len_alldata = len(alldata)
    self._check_data_length(names, alldata)
    return ({name: alldata[i + offset] for i, name in enumerate(names) if i < len_alldata}, names)