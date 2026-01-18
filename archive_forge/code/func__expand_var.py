from __future__ import annotations
import collections
from typing import Any, TYPE_CHECKING, cast
from .charset import Charset
from .variable import Variable
def _expand_var(self, variable: Variable, value: Any) -> str | None:
    """Expand a single variable."""
    return self._encode_var(variable, self._uri_encode_name(variable.name), value, delim='.' if variable.explode else ',')