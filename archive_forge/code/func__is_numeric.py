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
@property
def _is_numeric(self) -> bool:
    """
        Whether columns with this dtype should be considered numeric.
        """
    return pa.types.is_integer(self.pyarrow_dtype) or pa.types.is_floating(self.pyarrow_dtype) or pa.types.is_decimal(self.pyarrow_dtype)