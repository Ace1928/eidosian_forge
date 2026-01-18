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
def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
    from pandas.core.dtypes.cast import find_common_type
    new_dtype = find_common_type([dtype.numpy_dtype if isinstance(dtype, ArrowDtype) else dtype for dtype in dtypes])
    if not isinstance(new_dtype, np.dtype):
        return None
    try:
        pa_dtype = pa.from_numpy_dtype(new_dtype)
        return type(self)(pa_dtype)
    except NotImplementedError:
        return None