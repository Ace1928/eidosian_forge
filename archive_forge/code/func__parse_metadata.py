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
def _parse_metadata(self) -> None:
    done = False
    while not done:
        self._cached_page = self._path_or_buf.read(self._page_length)
        if len(self._cached_page) <= 0:
            break
        if len(self._cached_page) != self._page_length:
            raise ValueError('Failed to read a meta data page from the SAS file.')
        done = self._process_page_meta()