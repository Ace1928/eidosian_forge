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
class FilterBinOp(BinOp):
    filter: tuple[Any, Any, Index] | None = None

    def __repr__(self) -> str:
        if self.filter is None:
            return 'Filter: Not Initialized'
        return pprint_thing(f'[Filter : [{self.filter[0]}] -> [{self.filter[1]}]')

    def invert(self) -> Self:
        """invert the filter"""
        if self.filter is not None:
            self.filter = (self.filter[0], self.generate_filter_op(invert=True), self.filter[2])
        return self

    def format(self):
        """return the actual filter format"""
        return [self.filter]

    def evaluate(self) -> Self | None:
        if not self.is_valid:
            raise ValueError(f'query term is not valid [{self}]')
        rhs = self.conform(self.rhs)
        values = list(rhs)
        if self.is_in_table:
            if self.op in ['==', '!='] and len(values) > self._max_selectors:
                filter_op = self.generate_filter_op()
                self.filter = (self.lhs, filter_op, Index(values))
                return self
            return None
        if self.op in ['==', '!=']:
            filter_op = self.generate_filter_op()
            self.filter = (self.lhs, filter_op, Index(values))
        else:
            raise TypeError(f'passing a filterable condition to a non-table indexer [{self}]')
        return self

    def generate_filter_op(self, invert: bool=False):
        if self.op == '!=' and (not invert) or (self.op == '==' and invert):
            return lambda axis, vals: ~axis.isin(vals)
        else:
            return lambda axis, vals: axis.isin(vals)