from __future__ import annotations
import collections
from typing import Any, TYPE_CHECKING, cast
from .charset import Charset
from .variable import Variable
class FormStyleQueryExpansion(PathStyleExpansion):
    """
    Form-Style Query Expansion {?var}.

    https://tools.ietf.org/html/rfc6570#section-3.2.8
    """
    operator = '?'
    partial_operator = '&'
    output_prefix = '?'
    var_joiner = '&'
    partial_joiner = '&'

    def __init__(self, variables: str) -> None:
        super().__init__(variables)

    def _expand_var(self, variable: Variable, value: Any) -> str | None:
        """Expand a single variable."""
        if variable.explode:
            return self._encode_var(variable, self._uri_encode_name(variable.name), value, delim='&')
        value = self._encode_var(variable, self._uri_encode_name(variable.name), value, delim=',')
        return self._uri_encode_name(variable.name) + '=' + value if value is not None else None