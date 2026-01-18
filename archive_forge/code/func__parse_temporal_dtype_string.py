from __future__ import annotations
from datetime import (
from decimal import Decimal
import re
import numpy as np
from pandas._libs.tslibs import (
from pandas._typing import (
from pandas.compat import pa_version_under7p0
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.base import (
from pandas.core.dtypes.dtypes import CategoricalDtypeType
@classmethod
def _parse_temporal_dtype_string(cls, string: str) -> ArrowDtype:
    """
        Construct a temporal ArrowDtype from string.
        """
    head, tail = string.split('[', 1)
    if not tail.endswith(']'):
        raise ValueError
    tail = tail[:-1]
    if head == 'timestamp':
        assert ',' in tail
        unit, tz = tail.split(',', 1)
        unit = unit.strip()
        tz = tz.strip()
        if tz.startswith('tz='):
            tz = tz[3:]
        pa_type = pa.timestamp(unit, tz=tz)
        dtype = cls(pa_type)
        return dtype
    raise NotImplementedError(string)