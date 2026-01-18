from __future__ import annotations
from contextlib import suppress
import copy
from datetime import (
import itertools
import os
import re
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.lib import is_string_array
from pandas._libs.tslibs import timezones
from pandas.compat._optional import import_optional_dependency
from pandas.compat.pickle_compat import patch_pickle
from pandas.errors import (
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import array_equivalent
from pandas import (
from pandas.core.arrays import (
import pandas.core.common as com
from pandas.core.computation.pytables import (
from pandas.core.construction import extract_array
from pandas.core.indexes.api import ensure_index
from pandas.core.internals import (
from pandas.io.common import stringify_path
from pandas.io.formats.printing import (
def _dtype_to_kind(dtype_str: str) -> str:
    """
    Find the "kind" string describing the given dtype name.
    """
    dtype_str = _ensure_decoded(dtype_str)
    if dtype_str.startswith(('string', 'bytes')):
        kind = 'string'
    elif dtype_str.startswith('float'):
        kind = 'float'
    elif dtype_str.startswith('complex'):
        kind = 'complex'
    elif dtype_str.startswith(('int', 'uint')):
        kind = 'integer'
    elif dtype_str.startswith('datetime64'):
        kind = dtype_str
    elif dtype_str.startswith('timedelta'):
        kind = 'timedelta64'
    elif dtype_str.startswith('bool'):
        kind = 'bool'
    elif dtype_str.startswith('category'):
        kind = 'category'
    elif dtype_str.startswith('period'):
        kind = 'integer'
    elif dtype_str == 'object':
        kind = 'object'
    else:
        raise ValueError(f'cannot interpret dtype of [{dtype_str}]')
    return kind