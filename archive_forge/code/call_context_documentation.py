from contextvars import Context, ContextVar, copy_context
from typing import Any, Dict, List
Get the map of names to their values from the _NAME_VALUE_MAP context var.

        If the map does not exist in the current context, an empty map is created and returned.
        