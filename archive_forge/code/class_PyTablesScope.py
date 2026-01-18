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
class PyTablesScope(_scope.Scope):
    __slots__ = ('queryables',)
    queryables: dict[str, Any]

    def __init__(self, level: int, global_dict=None, local_dict=None, queryables: dict[str, Any] | None=None) -> None:
        super().__init__(level + 1, global_dict=global_dict, local_dict=local_dict)
        self.queryables = queryables or {}