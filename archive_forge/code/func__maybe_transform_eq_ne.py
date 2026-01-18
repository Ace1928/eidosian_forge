from __future__ import annotations
import ast
from functools import (
from keyword import iskeyword
import tokenize
from typing import (
import numpy as np
from pandas.errors import UndefinedVariableError
import pandas.core.common as com
from pandas.core.computation.ops import (
from pandas.core.computation.parsing import (
from pandas.core.computation.scope import Scope
from pandas.io.formats import printing
def _maybe_transform_eq_ne(self, node, left=None, right=None):
    if left is None:
        left = self.visit(node.left, side='left')
    if right is None:
        right = self.visit(node.right, side='right')
    op, op_class, left, right = self._rewrite_membership_op(node, left, right)
    return (op, op_class, left, right)