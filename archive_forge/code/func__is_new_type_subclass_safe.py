import inspect
import sys
from datetime import datetime, timezone
from collections import Counter
from typing import (Collection, Mapping, Optional, TypeVar, Any, Type, Tuple,
def _is_new_type_subclass_safe(cls, classinfo):
    super_type = getattr(cls, '__supertype__', None)
    if super_type:
        return _is_new_type_subclass_safe(super_type, classinfo)
    try:
        return issubclass(cls, classinfo)
    except Exception:
        return False