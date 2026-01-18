from __future__ import annotations
import codecs
from functools import wraps
import re
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._typing import (
from pandas.util._decorators import Appender
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import isna
from pandas.core.arrays import ExtensionArray
from pandas.core.base import NoNewAttributesMixin
from pandas.core.construction import extract_array
def _get_group_names(regex: re.Pattern) -> list[Hashable]:
    """
    Get named groups from compiled regex.

    Unnamed groups are numbered.

    Parameters
    ----------
    regex : compiled regex

    Returns
    -------
    list of column labels
    """
    names = {v: k for k, v in regex.groupindex.items()}
    return [names.get(1 + i, i) for i in range(regex.groups)]