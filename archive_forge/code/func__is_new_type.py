import inspect
import sys
from datetime import datetime, timezone
from collections import Counter
from typing import (Collection, Mapping, Optional, TypeVar, Any, Type, Tuple,
def _is_new_type(type_):
    return inspect.isfunction(type_) and hasattr(type_, '__supertype__')