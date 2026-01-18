from __future__ import annotations
import ast
from decimal import (
from functools import partial
from typing import (
import numpy as np
from pandas._libs.tslibs import (
from pandas.errors import UndefinedVariableError
from pandas.core.dtypes.common import is_list_like
import pandas.core.common as com
from pandas.core.computation import (
from pandas.core.computation.common import ensure_decoded
from pandas.core.computation.expr import BaseExprVisitor
from pandas.core.computation.ops import is_term
from pandas.core.construction import extract_array
from pandas.core.indexes.base import Index
from pandas.io.formats.printing import (
class TermValue:
    """hold a term value the we use to construct a condition/filter"""

    def __init__(self, value, converted, kind: str) -> None:
        assert isinstance(kind, str), kind
        self.value = value
        self.converted = converted
        self.kind = kind

    def tostring(self, encoding) -> str:
        """quote the string if not encoded else encode and return"""
        if self.kind == 'string':
            if encoding is not None:
                return str(self.converted)
            return f'"{self.converted}"'
        elif self.kind == 'float':
            return repr(self.converted)
        return str(self.converted)