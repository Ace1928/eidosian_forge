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
def _create_nodes_and_group(self, key: str) -> Node:
    """Create nodes from key and return group name."""
    assert self._handle is not None
    paths = key.split('/')
    path = '/'
    for p in paths:
        if not len(p):
            continue
        new_path = path
        if not path.endswith('/'):
            new_path += '/'
        new_path += p
        group = self.get_node(new_path)
        if group is None:
            group = self._handle.create_group(path, p)
        path = new_path
    return group