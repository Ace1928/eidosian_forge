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
def _validate_left_right_on(self, left_on, right_on):
    left_on, right_on = super()._validate_left_right_on(left_on, right_on)
    if len(left_on) != 1 and (not self.left_index):
        raise MergeError('can only asof on a key for left')
    if len(right_on) != 1 and (not self.right_index):
        raise MergeError('can only asof on a key for right')
    if self.left_index and isinstance(self.left.index, MultiIndex):
        raise MergeError('left can only have one index')
    if self.right_index and isinstance(self.right.index, MultiIndex):
        raise MergeError('right can only have one index')
    if self.by is not None:
        if self.left_by is not None or self.right_by is not None:
            raise MergeError('Can only pass by OR left_by and right_by')
        self.left_by = self.right_by = self.by
    if self.left_by is None and self.right_by is not None:
        raise MergeError('missing left_by')
    if self.left_by is not None and self.right_by is None:
        raise MergeError('missing right_by')
    if not self.left_index:
        left_on_0 = left_on[0]
        if isinstance(left_on_0, _known):
            lo_dtype = left_on_0.dtype
        else:
            lo_dtype = self.left._get_label_or_level_values(left_on_0).dtype if left_on_0 in self.left.columns else self.left.index.get_level_values(left_on_0)
    else:
        lo_dtype = self.left.index.dtype
    if not self.right_index:
        right_on_0 = right_on[0]
        if isinstance(right_on_0, _known):
            ro_dtype = right_on_0.dtype
        else:
            ro_dtype = self.right._get_label_or_level_values(right_on_0).dtype if right_on_0 in self.right.columns else self.right.index.get_level_values(right_on_0)
    else:
        ro_dtype = self.right.index.dtype
    if is_object_dtype(lo_dtype) or is_object_dtype(ro_dtype) or is_string_dtype(lo_dtype) or is_string_dtype(ro_dtype):
        raise MergeError(f'Incompatible merge dtype, {repr(ro_dtype)} and {repr(lo_dtype)}, both sides must have numeric dtype')
    if self.left_by is not None:
        if not is_list_like(self.left_by):
            self.left_by = [self.left_by]
        if not is_list_like(self.right_by):
            self.right_by = [self.right_by]
        if len(self.left_by) != len(self.right_by):
            raise MergeError('left_by and right_by must be the same length')
        left_on = self.left_by + list(left_on)
        right_on = self.right_by + list(right_on)
    return (left_on, right_on)