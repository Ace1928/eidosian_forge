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
def _write_map(self) -> None:
    """
        Called twice during file write. The first populates the values in
        the map with 0s.  The second call writes the final map locations when
        all blocks have been written.
        """
    if not self._map:
        self._map = {'stata_data': 0, 'map': self.handles.handle.tell(), 'variable_types': 0, 'varnames': 0, 'sortlist': 0, 'formats': 0, 'value_label_names': 0, 'variable_labels': 0, 'characteristics': 0, 'data': 0, 'strls': 0, 'value_labels': 0, 'stata_data_close': 0, 'end-of-file': 0}
    self.handles.handle.seek(self._map['map'])
    bio = BytesIO()
    for val in self._map.values():
        bio.write(struct.pack(self._byteorder + 'Q', val))
    self._write_bytes(self._tag(bio.getvalue(), 'map'))