import inspect
import sys
from datetime import datetime, timezone
from collections import Counter
from typing import (Collection, Mapping, Optional, TypeVar, Any, Type, Tuple,
def _is_counter(type_):
    return _issubclass_safe(_get_type_origin(type_), Counter)