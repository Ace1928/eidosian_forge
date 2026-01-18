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
def _build_elems(self, d: dict[str, Any], elem_row: Any) -> None:
    """
        Create child elements of row.

        This method adds child elements using elem_cols to row element and
        works with tuples for multindex or hierarchical columns.
        """
    sub_element_cls = self._sub_element_cls
    if not self.elem_cols:
        return
    for col in self.elem_cols:
        elem_name = self._get_flat_col_name(col)
        try:
            val = None if isna(d[col]) or d[col] == '' else str(d[col])
            sub_element_cls(elem_row, elem_name).text = val
        except KeyError:
            raise KeyError(f'no valid column, {col}')