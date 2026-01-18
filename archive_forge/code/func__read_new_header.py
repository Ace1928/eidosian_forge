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
def _read_new_header(self) -> None:
    self._path_or_buf.read(27)
    self._format_version = int(self._path_or_buf.read(3))
    if self._format_version not in [117, 118, 119]:
        raise ValueError(_version_error.format(version=self._format_version))
    self._set_encoding()
    self._path_or_buf.read(21)
    self._byteorder = '>' if self._path_or_buf.read(3) == b'MSF' else '<'
    self._path_or_buf.read(15)
    self._nvar = self._read_uint16() if self._format_version <= 118 else self._read_uint32()
    self._path_or_buf.read(7)
    self._nobs = self._get_nobs()
    self._path_or_buf.read(11)
    self._data_label = self._get_data_label()
    self._path_or_buf.read(19)
    self._time_stamp = self._get_time_stamp()
    self._path_or_buf.read(26)
    self._path_or_buf.read(8)
    self._path_or_buf.read(8)
    self._seek_vartypes = self._read_int64() + 16
    self._seek_varnames = self._read_int64() + 10
    self._seek_sortlist = self._read_int64() + 10
    self._seek_formats = self._read_int64() + 9
    self._seek_value_label_names = self._read_int64() + 19
    self._seek_variable_labels = self._get_seek_variable_labels()
    self._path_or_buf.read(8)
    self._data_location = self._read_int64() + 6
    self._seek_strls = self._read_int64() + 7
    self._seek_value_labels = self._read_int64() + 14
    self._typlist, self._dtyplist = self._get_dtypes(self._seek_vartypes)
    self._path_or_buf.seek(self._seek_varnames)
    self._varlist = self._get_varlist()
    self._path_or_buf.seek(self._seek_sortlist)
    self._srtlist = self._read_int16_count(self._nvar + 1)[:-1]
    self._path_or_buf.seek(self._seek_formats)
    self._fmtlist = self._get_fmtlist()
    self._path_or_buf.seek(self._seek_value_label_names)
    self._lbllist = self._get_lbllist()
    self._path_or_buf.seek(self._seek_variable_labels)
    self._variable_labels = self._get_variable_labels()