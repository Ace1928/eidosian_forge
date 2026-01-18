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
def _validate_where(w):
    """
    Validate that the where statement is of the right type.

    The type may either be String, Expr, or list-like of Exprs.

    Parameters
    ----------
    w : String term expression, Expr, or list-like of Exprs.

    Returns
    -------
    where : The original where clause if the check was successful.

    Raises
    ------
    TypeError : An invalid data type was passed in for w (e.g. dict).
    """
    if not (isinstance(w, (PyTablesExpr, str)) or is_list_like(w)):
        raise TypeError('where must be passed as a string, PyTablesExpr, or list-like of PyTablesExpr')
    return w