from __future__ import annotations
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.astype import astype_array
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas.core.dtypes.generic import (
def _get_result_dtype(to_concat: Sequence[ArrayLike], non_empties: Sequence[ArrayLike]) -> tuple[bool, set[str], DtypeObj | None]:
    target_dtype = None
    dtypes = {obj.dtype for obj in to_concat}
    kinds = {obj.dtype.kind for obj in to_concat}
    any_ea = any((not isinstance(x, np.ndarray) for x in to_concat))
    if any_ea:
        if len(dtypes) != 1:
            target_dtype = find_common_type([x.dtype for x in to_concat])
            target_dtype = common_dtype_categorical_compat(to_concat, target_dtype)
    elif not len(non_empties):
        if len(kinds) != 1:
            if not len(kinds - {'i', 'u', 'f'}) or not len(kinds - {'b', 'i', 'u'}):
                pass
            else:
                target_dtype = np.dtype(object)
                kinds = {'o'}
    else:
        target_dtype = np_find_common_type(*dtypes)
    return (any_ea, kinds, target_dtype)