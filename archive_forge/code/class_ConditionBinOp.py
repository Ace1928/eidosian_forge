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
class ConditionBinOp(BinOp):

    def __repr__(self) -> str:
        return pprint_thing(f'[Condition : [{self.condition}]]')

    def invert(self):
        """invert the condition"""
        raise NotImplementedError('cannot use an invert condition when passing to numexpr')

    def format(self):
        """return the actual ne format"""
        return self.condition

    def evaluate(self) -> Self | None:
        if not self.is_valid:
            raise ValueError(f'query term is not valid [{self}]')
        if not self.is_in_table:
            return None
        rhs = self.conform(self.rhs)
        values = [self.convert_value(v) for v in rhs]
        if self.op in ['==', '!=']:
            if len(values) <= self._max_selectors:
                vs = [self.generate(v) for v in values]
                self.condition = f'({' | '.join(vs)})'
            else:
                return None
        else:
            self.condition = self.generate(values[0])
        return self