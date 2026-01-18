from __future__ import annotations
from collections.abc import (
import datetime
from functools import partial
from typing import (
import uuid
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.lib import is_range_indexer
from pandas._typing import (
from pandas.errors import MergeError
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas import (
import pandas.core.algorithms as algos
from pandas.core.arrays import (
from pandas.core.arrays.string_ import StringDtype
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.frame import _merge_doc
from pandas.core.indexes.api import default_index
from pandas.core.sorting import (
@final
def _validate_validate_kwd(self, validate: str) -> None:
    if self.left_index:
        left_unique = self.orig_left.index.is_unique
    else:
        left_unique = MultiIndex.from_arrays(self.left_join_keys).is_unique
    if self.right_index:
        right_unique = self.orig_right.index.is_unique
    else:
        right_unique = MultiIndex.from_arrays(self.right_join_keys).is_unique
    if validate in ['one_to_one', '1:1']:
        if not left_unique and (not right_unique):
            raise MergeError('Merge keys are not unique in either left or right dataset; not a one-to-one merge')
        if not left_unique:
            raise MergeError('Merge keys are not unique in left dataset; not a one-to-one merge')
        if not right_unique:
            raise MergeError('Merge keys are not unique in right dataset; not a one-to-one merge')
    elif validate in ['one_to_many', '1:m']:
        if not left_unique:
            raise MergeError('Merge keys are not unique in left dataset; not a one-to-many merge')
    elif validate in ['many_to_one', 'm:1']:
        if not right_unique:
            raise MergeError('Merge keys are not unique in right dataset; not a many-to-one merge')
    elif validate in ['many_to_many', 'm:m']:
        pass
    else:
        raise ValueError(f'"{validate}" is not a valid argument. Valid arguments are:\n- "1:1"\n- "1:m"\n- "m:1"\n- "m:m"\n- "one_to_one"\n- "one_to_many"\n- "many_to_one"\n- "many_to_many"')