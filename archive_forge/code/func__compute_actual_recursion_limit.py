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
def _compute_actual_recursion_limit(self):
    recursion_limit = sys.getrecursionlimit() - get_stack_depth() - 2 * RECURSION_LIMIT
    if recursion_limit <= RECURSION_LIMIT:
        self.recursion_stack = []
        raise RevertToNonrecursive()
    return recursion_limit