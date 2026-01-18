from __future__ import annotations
from datetime import (
from decimal import Decimal
import re
from typing import (
import warnings
import numpy as np
import pytz
from pandas._libs import (
from pandas._libs.interval import Interval
from pandas._libs.properties import cache_readonly
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import (
from pandas._libs.tslibs.offsets import BDay
from pandas.compat import pa_version_under10p1
from pandas.errors import PerformanceWarning
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import (
from pandas.util import capitalize_first_letter
class PandasExtensionDtype(ExtensionDtype):
    """
    A np.dtype duck-typed class, suitable for holding a custom dtype.

    THIS IS NOT A REAL NUMPY DTYPE
    """
    type: Any
    kind: Any
    subdtype = None
    str: str_type
    num = 100
    shape: tuple[int, ...] = ()
    itemsize = 8
    base: DtypeObj | None = None
    isbuiltin = 0
    isnative = 0
    _cache_dtypes: dict[str_type, PandasExtensionDtype] = {}

    def __repr__(self) -> str_type:
        """
        Return a string representation for a particular object.
        """
        return str(self)

    def __hash__(self) -> int:
        raise NotImplementedError('sub-classes should implement an __hash__ method')

    def __getstate__(self) -> dict[str_type, Any]:
        return {k: getattr(self, k, None) for k in self._metadata}

    @classmethod
    def reset_cache(cls) -> None:
        """clear the cache"""
        cls._cache_dtypes = {}