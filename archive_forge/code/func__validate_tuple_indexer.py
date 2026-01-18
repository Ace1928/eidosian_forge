from __future__ import annotations
from contextlib import suppress
import sys
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs.indexing import NDFrameIndexerBase
from pandas._libs.lib import item_from_zerodim
from pandas.compat import PYPY
from pandas.errors import (
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core import algorithms as algos
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.indexers import (
from pandas.core.indexes.api import (
@final
def _validate_tuple_indexer(self, key: tuple) -> tuple:
    """
        Check the key for valid keys across my indexer.
        """
    key = self._validate_key_length(key)
    key = self._expand_ellipsis(key)
    for i, k in enumerate(key):
        try:
            self._validate_key(k, i)
        except ValueError as err:
            raise ValueError(f'Location based indexing can only have [{self._valid_types}] types') from err
    return key