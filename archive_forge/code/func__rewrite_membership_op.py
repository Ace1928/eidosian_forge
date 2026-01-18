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
def _rewrite_membership_op(self, node, left, right):
    op_instance = node.op
    op_type = type(op_instance)
    if is_term(left) and is_term(right) and (op_type in self.rewrite_map):
        left_list, right_list = map(_is_list, (left, right))
        left_str, right_str = map(_is_str, (left, right))
        if left_list or right_list or left_str or right_str:
            op_instance = self.rewrite_map[op_type]()
        if right_str:
            name = self.env.add_tmp([right.value])
            right = self.term_type(name, self.env)
        if left_str:
            name = self.env.add_tmp([left.value])
            left = self.term_type(name, self.env)
    op = self.visit(op_instance)
    return (op, op_instance, left, right)