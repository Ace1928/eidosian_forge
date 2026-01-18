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
class JointConditionBinOp(ConditionBinOp):

    def evaluate(self) -> Self:
        self.condition = f'({self.lhs.condition} {self.op} {self.rhs.condition})'
        return self