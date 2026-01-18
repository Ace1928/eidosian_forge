import inspect
import logging
import sys
from copy import deepcopy
from collections import deque
from pyomo.common.deprecation import deprecated, deprecation_warning
from pyomo.common.errors import DeveloperError, TemplateExpressionError
from pyomo.common.numeric_types import (
import pyomo.core.expr.expr_common as common
from pyomo.core.expr.symbol_map import SymbolMap
def _recursive_frame_to_nonrecursive_stack(self, local):
    child_idx = local['child_idx']
    _arg_list = [None] * child_idx
    _arg_list.append(local['child'])
    _arg_list.extend(local['arg_iter'])
    if not self.recursion_stack:
        child_idx -= 1
    self.recursion_stack.append((local['node'], _arg_list, len(_arg_list) - 1, local['data'], child_idx))