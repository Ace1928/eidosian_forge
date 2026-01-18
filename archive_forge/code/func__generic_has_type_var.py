import sys
import types
import typing
import typing_extensions
from mypy_extensions import _TypedDictMeta as _TypedDictMeta_Mypy
def _generic_has_type_var(tp):
    if tp.__parameters__:
        for t in tp.__parameters__:
            if _has_type_var(t):
                return True
    return False