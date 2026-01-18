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
def _read_page_header(self) -> None:
    bit_offset = self._page_bit_offset
    tx = const.page_type_offset + bit_offset
    self._current_page_type = self._read_uint(tx, const.page_type_length) & const.page_type_mask2
    tx = const.block_count_offset + bit_offset
    self._current_page_block_count = self._read_uint(tx, const.block_count_length)
    tx = const.subheader_count_offset + bit_offset
    self._current_page_subheaders_count = self._read_uint(tx, const.subheader_count_length)