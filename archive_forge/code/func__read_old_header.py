from __future__ import annotations
from collections import abc
from datetime import (
from io import BytesIO
import os
import struct
import sys
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.lib import infer_dtype
from pandas._libs.writers import max_len_string_array
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas import (
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
from pandas.core.indexes.range import RangeIndex
from pandas.core.series import Series
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import get_handle
def _read_old_header(self, first_char: bytes) -> None:
    self._format_version = int(first_char[0])
    if self._format_version not in [104, 105, 108, 111, 113, 114, 115]:
        raise ValueError(_version_error.format(version=self._format_version))
    self._set_encoding()
    self._byteorder = '>' if self._read_int8() == 1 else '<'
    self._filetype = self._read_int8()
    self._path_or_buf.read(1)
    self._nvar = self._read_uint16()
    self._nobs = self._get_nobs()
    self._data_label = self._get_data_label()
    self._time_stamp = self._get_time_stamp()
    if self._format_version > 108:
        typlist = [int(c) for c in self._path_or_buf.read(self._nvar)]
    else:
        buf = self._path_or_buf.read(self._nvar)
        typlistb = np.frombuffer(buf, dtype=np.uint8)
        typlist = []
        for tp in typlistb:
            if tp in self.OLD_TYPE_MAPPING:
                typlist.append(self.OLD_TYPE_MAPPING[tp])
            else:
                typlist.append(tp - 127)
    try:
        self._typlist = [self.TYPE_MAP[typ] for typ in typlist]
    except ValueError as err:
        invalid_types = ','.join([str(x) for x in typlist])
        raise ValueError(f'cannot convert stata types [{invalid_types}]') from err
    try:
        self._dtyplist = [self.DTYPE_MAP[typ] for typ in typlist]
    except ValueError as err:
        invalid_dtypes = ','.join([str(x) for x in typlist])
        raise ValueError(f'cannot convert stata dtypes [{invalid_dtypes}]') from err
    if self._format_version > 108:
        self._varlist = [self._decode(self._path_or_buf.read(33)) for _ in range(self._nvar)]
    else:
        self._varlist = [self._decode(self._path_or_buf.read(9)) for _ in range(self._nvar)]
    self._srtlist = self._read_int16_count(self._nvar + 1)[:-1]
    self._fmtlist = self._get_fmtlist()
    self._lbllist = self._get_lbllist()
    self._variable_labels = self._get_variable_labels()
    if self._format_version > 104:
        while True:
            data_type = self._read_int8()
            if self._format_version > 108:
                data_len = self._read_int32()
            else:
                data_len = self._read_int16()
            if data_type == 0:
                break
            self._path_or_buf.read(data_len)
    self._data_location = self._path_or_buf.tell()