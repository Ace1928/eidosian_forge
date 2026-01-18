from __future__ import annotations
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.tslibs import is_supported_dtype
from pandas._typing import (
from pandas.compat import (
from pandas.errors import AbstractMethodError
from pandas.util._decorators import doc
from pandas.util._validators import validate_fillna_kwargs
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import BaseMaskedDtype
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.algorithms import (
from pandas.core.array_algos import (
from pandas.core.array_algos.quantile import quantile_with_mask
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._utils import to_numpy_dtype_inference
from pandas.core.arrays.base import ExtensionArray
from pandas.core.construction import (
from pandas.core.indexers import check_array_indexer
from pandas.core.ops import invalid_comparison
from pandas.core.util.hashing import hash_array
from pandas.compat.numpy import function as nv
@doc(ExtensionArray._values_for_argsort)
def _values_for_argsort(self) -> np.ndarray:
    return self._data