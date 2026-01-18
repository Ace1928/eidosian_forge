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
def _check_comments(self, lines: list[list[Scalar]]) -> list[list[Scalar]]:
    if self.comment is None:
        return lines
    ret = []
    for line in lines:
        rl = []
        for x in line:
            if not isinstance(x, str) or self.comment not in x or x in self.na_values:
                rl.append(x)
            else:
                x = x[:x.find(self.comment)]
                if len(x) > 0:
                    rl.append(x)
                break
        ret.append(rl)
    return ret