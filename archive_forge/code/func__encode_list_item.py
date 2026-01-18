from __future__ import annotations
import collections
from typing import Any, TYPE_CHECKING, cast
from .charset import Charset
from .variable import Variable
def _encode_list_item(self, variable: Variable, name: str, index: int, item: Any, delim: str, prefix: str, joiner: str, first: bool) -> str | None:
    """Encode a list item for a variable."""
    if variable.array:
        if name:
            prefix = prefix + '[' + name + ']' if prefix else name
        return self._encode_var(variable, str(index), item, delim, prefix, joiner, False)
    return self._encode_var(variable, name, item, delim, prefix, '=' if variable.explode else '.', False)