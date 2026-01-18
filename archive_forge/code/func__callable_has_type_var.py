import sys
import types
import typing
import typing_extensions
from mypy_extensions import _TypedDictMeta as _TypedDictMeta_Mypy
def _callable_has_type_var(tp):
    if tp.__args__:
        for t in tp.__args__:
            if _has_type_var(t):
                return True
    return _has_type_var(tp.__result__)