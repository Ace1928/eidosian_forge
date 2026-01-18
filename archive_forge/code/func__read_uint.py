from __future__ import annotations
from collections import abc
from datetime import (
import sys
from typing import TYPE_CHECKING
import numpy as np
from pandas._libs.byteswap import (
from pandas._libs.sas import (
from pandas._libs.tslibs.conversion import cast_from_unit_vectorized
from pandas.errors import EmptyDataError
import pandas as pd
from pandas import (
from pandas.io.common import get_handle
import pandas.io.sas.sas_constants as const
from pandas.io.sas.sasreader import ReaderBase
def _read_uint(self, offset: int, width: int) -> int:
    assert self._cached_page is not None
    if width == 1:
        return self._read_bytes(offset, 1)[0]
    elif width == 2:
        return read_uint16_with_byteswap(self._cached_page, offset, self.need_byteswap)
    elif width == 4:
        return read_uint32_with_byteswap(self._cached_page, offset, self.need_byteswap)
    elif width == 8:
        return read_uint64_with_byteswap(self._cached_page, offset, self.need_byteswap)
    else:
        self.close()
        raise ValueError('invalid int width')