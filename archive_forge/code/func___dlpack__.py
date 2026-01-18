from __future__ import annotations
from typing import (
from pandas.core.interchange.dataframe_protocol import (
def __dlpack__(self) -> Any:
    """
        Represent this structure as DLPack interface.
        """
    return self._x.__dlpack__()