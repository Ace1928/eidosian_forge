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
def _process_page_meta(self) -> bool:
    self._read_page_header()
    pt = const.page_meta_types + [const.page_amd_type, const.page_mix_type]
    if self._current_page_type in pt:
        self._process_page_metadata()
    is_data_page = self._current_page_type == const.page_data_type
    is_mix_page = self._current_page_type == const.page_mix_type
    return bool(is_data_page or is_mix_page or self._current_page_data_subheader_pointers != [])