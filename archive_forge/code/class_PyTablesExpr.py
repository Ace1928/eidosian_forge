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
class PyTablesExpr(expr.Expr):
    """
    Hold a pytables-like expression, comprised of possibly multiple 'terms'.

    Parameters
    ----------
    where : string term expression, PyTablesExpr, or list-like of PyTablesExprs
    queryables : a "kinds" map (dict of column name -> kind), or None if column
        is non-indexable
    encoding : an encoding that will encode the query terms

    Returns
    -------
    a PyTablesExpr object

    Examples
    --------
    'index>=date'
    "columns=['A', 'D']"
    'columns=A'
    'columns==A'
    "~(columns=['A','B'])"
    'index>df.index[3] & string="bar"'
    '(index>df.index[3] & index<=df.index[6]) | string="bar"'
    "ts>=Timestamp('2012-02-01')"
    "major_axis>=20130101"
    """
    _visitor: PyTablesExprVisitor | None
    env: PyTablesScope
    expr: str

    def __init__(self, where, queryables: dict[str, Any] | None=None, encoding=None, scope_level: int=0) -> None:
        where = _validate_where(where)
        self.encoding = encoding
        self.condition = None
        self.filter = None
        self.terms = None
        self._visitor = None
        local_dict: _scope.DeepChainMap[Any, Any] | None = None
        if isinstance(where, PyTablesExpr):
            local_dict = where.env.scope
            _where = where.expr
        elif is_list_like(where):
            where = list(where)
            for idx, w in enumerate(where):
                if isinstance(w, PyTablesExpr):
                    local_dict = w.env.scope
                else:
                    where[idx] = _validate_where(w)
            _where = ' & '.join([f'({w})' for w in com.flatten(where)])
        else:
            _where = where
        self.expr = _where
        self.env = PyTablesScope(scope_level + 1, local_dict=local_dict)
        if queryables is not None and isinstance(self.expr, str):
            self.env.queryables.update(queryables)
            self._visitor = PyTablesExprVisitor(self.env, queryables=queryables, parser='pytables', engine='pytables', encoding=encoding)
            self.terms = self.parse()

    def __repr__(self) -> str:
        if self.terms is not None:
            return pprint_thing(self.terms)
        return pprint_thing(self.expr)

    def evaluate(self):
        """create and return the numexpr condition and filter"""
        try:
            self.condition = self.terms.prune(ConditionBinOp)
        except AttributeError as err:
            raise ValueError(f'cannot process expression [{self.expr}], [{self}] is not a valid condition') from err
        try:
            self.filter = self.terms.prune(FilterBinOp)
        except AttributeError as err:
            raise ValueError(f'cannot process expression [{self.expr}], [{self}] is not a valid filter') from err
        return (self.condition, self.filter)