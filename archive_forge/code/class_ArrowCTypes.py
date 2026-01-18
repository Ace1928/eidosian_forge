from __future__ import annotations
import typing
import numpy as np
from pandas._libs import lib
from pandas.core.dtypes.dtypes import (
class ArrowCTypes:
    """
    Enum for Apache Arrow C type format strings.

    The Arrow C data interface:
    https://arrow.apache.org/docs/format/CDataInterface.html#data-type-description-format-strings
    """
    NULL = 'n'
    BOOL = 'b'
    INT8 = 'c'
    UINT8 = 'C'
    INT16 = 's'
    UINT16 = 'S'
    INT32 = 'i'
    UINT32 = 'I'
    INT64 = 'l'
    UINT64 = 'L'
    FLOAT16 = 'e'
    FLOAT32 = 'f'
    FLOAT64 = 'g'
    STRING = 'u'
    LARGE_STRING = 'U'
    DATE32 = 'tdD'
    DATE64 = 'tdm'
    TIMESTAMP = 'ts{resolution}:{tz}'
    TIME = 'tt{resolution}'