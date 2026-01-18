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
def _read_next_page(self):
    self._current_page_data_subheader_pointers = []
    self._cached_page = self._path_or_buf.read(self._page_length)
    if len(self._cached_page) <= 0:
        return True
    elif len(self._cached_page) != self._page_length:
        self.close()
        msg = f'failed to read complete page from file (read {len(self._cached_page):d} of {self._page_length:d} bytes)'
        raise ValueError(msg)
    self._read_page_header()
    if self._current_page_type in const.page_meta_types:
        self._process_page_metadata()
    if self._current_page_type not in const.page_meta_types + [const.page_data_type, const.page_mix_type]:
        return self._read_next_page()
    return False