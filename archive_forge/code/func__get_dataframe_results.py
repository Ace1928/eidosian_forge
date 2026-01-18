from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterable, List, Optional
def _get_dataframe_results(self, df: DataFrame) -> list:
    return list(map(self._convert_row_as_tuple, df.collect()))