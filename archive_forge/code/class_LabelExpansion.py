from __future__ import annotations
import collections
from typing import Any, TYPE_CHECKING, cast
from .charset import Charset
from .variable import Variable
class LabelExpansion(ExpressionExpansion):
    """
    Label Expansion with Dot-Prefix {.var}.

    https://tools.ietf.org/html/rfc6570#section-3.2.5
    """
    operator = '.'
    partial_operator = '.'
    output_prefix = '.'
    var_joiner = '.'
    partial_joiner = '.'

    def __init__(self, variables: str) -> None:
        super().__init__(variables[1:])

    def _expand_var(self, variable: Variable, value: Any) -> str | None:
        """Expand a single variable."""
        return self._encode_var(variable, self._uri_encode_name(variable.name), value, delim='.' if variable.explode else ',')