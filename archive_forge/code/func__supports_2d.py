from __future__ import annotations
from typing import (
import numpy as np
from pandas._libs import missing as libmissing
from pandas._libs.hashtable import object_hash
from pandas._libs.properties import cache_readonly
from pandas.errors import AbstractMethodError
from pandas.core.dtypes.generic import (
@property
def _supports_2d(self) -> bool:
    """
        Do ExtensionArrays with this dtype support 2D arrays?

        Historically ExtensionArrays were limited to 1D. By returning True here,
        authors can indicate that their arrays support 2D instances. This can
        improve performance in some cases, particularly operations with `axis=1`.

        Arrays that support 2D values should:

            - implement Array.reshape
            - subclass the Dim2CompatTests in tests.extension.base
            - _concat_same_type should support `axis` keyword
            - _reduce and reductions should support `axis` keyword
        """
    return False