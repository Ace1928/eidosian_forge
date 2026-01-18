from __future__ import annotations
import collections
from typing import Any, TYPE_CHECKING, cast
from .charset import Charset
from .variable import Variable
class FormStyleQueryContinuation(FormStyleQueryExpansion):
    """
    Form-Style Query Continuation {&var}.

    https://tools.ietf.org/html/rfc6570#section-3.2.9
    """
    operator = '&'
    output_prefix = '&'

    def __init__(self, variables: str) -> None:
        super().__init__(variables)