import inspect
import sys
from datetime import datetime, timezone
from collections import Counter
from typing import (Collection, Mapping, Optional, TypeVar, Any, Type, Tuple,
def _is_optional(type_):
    return _issubclass_safe(type_, Optional) or _hasargs(type_, type(None)) or type_ is Any