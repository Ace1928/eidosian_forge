from __future__ import annotations
import codecs
import io
from typing import (
import warnings
from pandas.errors import AbstractMethodError
from pandas.util._decorators import (
from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.missing import isna
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import get_handle
from pandas.io.xml import (
@final
def _handle_indexes(self) -> None:
    """
        Handle indexes.

        This method will add indexes into attr_cols or elem_cols.
        """
    if not self.index:
        return
    first_key = next(iter(self.frame_dicts))
    indexes: list[str] = [x for x in self.frame_dicts[first_key].keys() if x not in self.orig_cols]
    if self.attr_cols:
        self.attr_cols = indexes + self.attr_cols
    if self.elem_cols:
        self.elem_cols = indexes + self.elem_cols