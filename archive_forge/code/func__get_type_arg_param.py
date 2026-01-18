import inspect
import sys
from datetime import datetime, timezone
from collections import Counter
from typing import (Collection, Mapping, Optional, TypeVar, Any, Type, Tuple,
def _get_type_arg_param(tp: Type, index: int) -> Union[Type, _NoArgs]:
    _args = _get_type_args(tp)
    if _args is not _NO_ARGS:
        try:
            return cast(Tuple[Type, ...], _args)[index]
        except (TypeError, IndexError, NotImplementedError):
            pass
    return _NO_ARGS