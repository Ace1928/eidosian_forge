import sys
import types
import typing
import typing_extensions
from mypy_extensions import _TypedDictMeta as _TypedDictMeta_Mypy
def _tuple_has_type_var(tp):
    if tp.__tuple_params__:
        for t in tp.__tuple_params__:
            if _has_type_var(t):
                return True
    return False